import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve


def write_json(obj: Dict | List, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return json.dump(obj, f, indent=4)


def read_json(fpath: str) -> Dict | List:
    with open(fpath, 'r') as f:
        return json.load(f)


def trans(input_file, tokenizer, prompt_size):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    tokens = tokenizer.encode(text, add_special_tokens=True)
    data = []

    for i in range(0, len(tokens) - prompt_size + 1, prompt_size):
        prompt_tokens = tokens[i : i + prompt_size]
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)

        data.append({"prompt": prompt_text})
    
    return data


@torch.no_grad()
def mia(model, tokenizer, data, dataset_name):
    
    tokenizer.padding_side = 'right'
    scores = defaultdict(list)
    
    for i, d in enumerate(tqdm(data, total=len(data))):
        text = d['prompt']
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        if torch.isnan(loss):
            continue
        ll = -loss.item()  # log-likelihood

        scores['loss'].append(ll)

        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1) 
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        
        # mink
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.cpu().float().numpy())[:k_length]
            scores[f'mink_{ratio}'].append(np.mean(topk).item())
        # mink++
        mink_plus = (token_log_probs - mu) / (sigma.sqrt() + 1e-6)
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.cpu().float().numpy())[:k_length]
            scores[f'mink++_{ratio}'].append(np.nanmean(topk).item())

    output_result = {
        'loss': np.mean(scores['loss']),
        'mink40': np.mean(scores['mink++_0.4']),
        'results': scores
    }

    return output_result


def compute_auroc(scores1, scores2):
    """Compute AUROC between two sets of scores."""
    labels = [0] * len(scores1) + [1] * len(scores2)
    scores = scores1 + scores2
    fpr, tpr, _ = get_roc_curve(labels, scores)
    return get_auc(fpr, tpr)


def mia_eval_score(cfg, unlearn_times, model, tokenizer):
    
    prompt_size = 1024
    
    format_names = ["Chronological", "Interview", "Feature_Story", "Inverted_Pyramid", "Listicle"]
    forget_data = trans("data/Prof/Chronological.txt", tokenizer, prompt_size)
    retain_data = []
    for format_name in format_names[1:]: 
        retain_data.extend(trans(f"data/Prof/{format_name}.txt", tokenizer, prompt_size))
    holdout_data = trans(f"data/Prof/eval/holdout.txt", tokenizer, prompt_size)
    
    forget_results = mia(model, tokenizer, forget_data, dataset_name="forget")
    retain_results = mia(model, tokenizer, retain_data, dataset_name="retain")
    holdout_results = mia(model, tokenizer, holdout_data, dataset_name="holdout")
    
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')
    os.makedirs(curr_eval_dir, exist_ok=True)

    # Extract model type from model path
    match = re.search(r'-(\w+)$', cfg.model_path) 
    model_type = match.group(1) if match else "unknown"

    # Save MIA results
    write_json(forget_results, os.path.join(curr_eval_dir, f"mia/MIA_{model_type}_forget.json"))
    write_json(retain_results, os.path.join(curr_eval_dir, f"mia/MIA_{model_type}_retain.json"))
    write_json(holdout_results, os.path.join(curr_eval_dir, f"mia/MIA_{model_type}_holdout.json"))

    # Get all MinK++ ratios
    minkpp_keys = [key for key in forget_results["results"] if "mink" in key]

    # Compute AUROC for every MinK++ ratio
    auroc_results = {}
    for key in minkpp_keys:
        auroc_results[f"forget_retain_{key}"] = compute_auroc(forget_results["results"][key], retain_results["results"][key])
        auroc_results[f"retain_forget_{key}"] = compute_auroc(retain_results["results"][key], forget_results["results"][key])
        
        auroc_results[f"forget_holdout_{key}"] = compute_auroc(forget_results["results"][key], holdout_results["results"][key])
        auroc_results[f"holdout_forget{key}"] = compute_auroc(holdout_results["results"][key], forget_results["results"][key])
        
        auroc_results[f"retain_holdout_{key}"] = compute_auroc(retain_results["results"][key], holdout_results["results"][key])
        auroc_results[f"holdout_retain{key}"] = compute_auroc(holdout_results["results"][key], retain_results["results"][key])

    # Save AUROC results
    write_json(auroc_results, os.path.join(curr_eval_dir, f"mia/MIA_{model_type}_AUROC.json"))

    return forget_results, retain_results, holdout_results, auroc_results

