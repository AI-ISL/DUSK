import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(model, ref_model, inputs, loss_type, beta=0.1):
    if 'RMU' in loss_type:
        # you can choose the layer to apply RMU loss
        layer_id = 7  
        layer_module_updated = model.model.layers[layer_id]
        layer_module_frozen  = ref_model.model.layers[layer_id]

        forget_loss, regularization_loss = rmu_loss(
            model=model,
            ref_model=ref_model,
            inputs=inputs,
            layer_module_updated=layer_module_updated,
            layer_module_frozen=layer_module_frozen,
            steering_coeff=20,
            alpha=1 
        )
        return forget_loss, regularization_loss
    
    if 'SGA' in loss_type:
        forget_loss = sga_loss(
            model, 
            inputs,
            threshold=0.70, 
            top_k=1
        )
        regularization_loss = 0
        return forget_loss, regularization_loss

    # forget_loss
    if 'GA' in loss_type:
        forget_loss = ga_loss(model, inputs)
    elif 'NPO' in loss_type:
        forget_loss = npo_loss(model, ref_model, inputs, beta=beta)
    elif 'NONE' in loss_type:
        forget_loss = 0
    elif 'TV' in loss_type:
        forget_loss = reinforce_gd_loss(model, inputs)

    # regularization_loss
    if 'GD' in loss_type:
        regularization_loss = gd_loss(model, inputs)
    else:
        regularization_loss = 0
    

    return forget_loss, regularization_loss


def ga_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss


def npo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels,
                                attention_mask=attention_mask)
        loss_ref = get_batch_loss(ref_outputs.logits, labels)

    neg_log_ratios = loss_current - loss_ref
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    return loss

# Regularization Loss: GD
def gd_loss(model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss


def get_batch_loss(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


def me_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=None, attention_mask=attention_mask)
    loss = get_me_loss(outputs.logits, labels)

    return loss


def get_me_loss(logits, labels):
    num_labels = logits.shape[-1]

    assert logits.shape[:-1] == labels.shape, "Logits and labels must have compatible shapes."

    # Adjust logits and labels to exclude the last token
    labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)

    soft_outputs = F.softmax(logits, dim=-1).view(-1, num_labels)  # (bs*seq_len, vocab_size)
    uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels).to(logits.device)  # (bs*seq_len, vocab_size)

    loss_mask = (labels != -100).view(-1)  # (bs*(seq_len - 1))

    kl_div = F.kl_div((soft_outputs + 1e-12).log(), uniform_dist, reduction='none').sum(-1)  # (bs*(seq_len - 1))

    masked_kl_div = kl_div * loss_mask  # (bs*(seq_len - 1))
    loss = masked_kl_div.sum() / loss_mask.sum()

    return loss

# Reinforce model: Overfitting Forget Set
def reinforce_gd_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def rmu_loss(model, 
             ref_model, 
             inputs, 
             layer_module_updated,   
             layer_module_frozen,   
             steering_coeff=20.0,    
             alpha=100.0):           

    forget_input_ids, forget_labels, forget_attention_mask = inputs[0]
    retain_input_ids, retain_labels, retain_attention_mask = inputs[1]

    updated_forget_activations = forward_activation(
        model,
        forget_input_ids,
        forget_attention_mask,
        layer_module_updated
    )
    # updated_forget_activations: [batch_size, seq_len, hidden_dim]
    
    hidden_dim = updated_forget_activations.shape[-1]

    random_vec = torch.rand(
        (1, 1, hidden_dim),
        device=updated_forget_activations.device,
        dtype=updated_forget_activations.dtype  
    )

    rand_fp32 = random_vec.float()
    normed = rand_fp32 / rand_fp32.norm() * steering_coeff
    random_vec = normed.to(updated_forget_activations.dtype)

    forget_loss = F.mse_loss(updated_forget_activations, random_vec)

    updated_retain_activations = forward_activation(
        model,
        retain_input_ids,
        retain_attention_mask,
        layer_module_updated
    )
    frozen_retain_activations = forward_activation(
        ref_model,
        retain_input_ids,
        retain_attention_mask,
        layer_module_frozen,
        requires_grad=False
    )

    retain_loss = F.mse_loss(updated_retain_activations, frozen_retain_activations)

    total_forget_loss = forget_loss
    total_regularization_loss = alpha * retain_loss

    return total_forget_loss, total_regularization_loss

def forward_activation(model, input_ids, attention_mask, layer_module, requires_grad=True):
    cache = []

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0] 
        cache.append(out)

    handle = layer_module.register_forward_hook(hook_fn)
    if requires_grad:
        _ = model(input_ids, attention_mask=attention_mask)
    else:
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    handle.remove()

    return cache[0]

def sga_loss(model, inputs, threshold=0.7, top_k=1):
    forget_inputs = inputs[0]  # (input_ids, labels, attn)
    input_ids, labels, attention_mask = forget_inputs

    # forward
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits  # (B=batch size, T=sequece length, V=vocabulary size)

    # log_probs
    log_probs = F.log_softmax(logits, dim=-1) 
    gather_logp = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
    
    valid_mask = (labels != -100)
    sum_prob = torch.exp(gather_logp) * valid_mask
    avg_prob = sum_prob.sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)                                                                      

    # threshold
    chosen_mask = (avg_prob >= threshold).float() 
    if chosen_mask.sum() < 1:  
        # top_k
        sorted_indices = torch.argsort(avg_prob, descending=True)
        chosen_mask = torch.zeros_like(avg_prob)
        chosen_mask[sorted_indices[:top_k]] = 1.0

    # CrossEntropy (token-wise, reduction='none') 
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    token_loss = F.cross_entropy(
        shifted_logits.transpose(1,2), shifted_labels,
        ignore_index=-100, reduction='none'
    ) 

    # summation
    sample_loss = token_loss.sum(dim=1) 

    # mean of chosen_mask
    numerator   = (sample_loss * chosen_mask).sum()
    denominator = chosen_mask.sum().clamp_min(1.0)
    final_ce = numerator / denominator

    # GA => -CE
    loss = -final_ce
    return loss