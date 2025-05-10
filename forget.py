import os
import shutil
import warnings
from pathlib import Path

import hydra
import torch
import transformers
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from dataset import custom_data_collator_forget, ProfDataset, DefaultDataset
from trainer import CustomTrainerForgetting
from utils import get_model_identifiers_from_yaml, set_random_seed

warnings.filterwarnings('ignore')

matching = []

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_task_data(curr_save_dir, tokenizer, num_formats):
    local_rank = int(os.environ['LOCAL_RANK'])         
    format_names = ["Chronological", "Interview", "Feature_Story", "Inverted_Pyramid", "Listicle"]
    
    forget_data = DefaultDataset()
    retain_data = DefaultDataset()
    
    forget_data = DefaultDataset(f"data/Prof/{format_names[0]}.txt", tokenizer)
    for i in range(1, num_formats):
        retain_data += DefaultDataset(f"data/Prof/{format_names[i]}.txt", tokenizer)

    # if local_rank == 0:
    #     curr_data_path = os.path.join(curr_save_dir, 'task_data')
    #     os.makedirs(curr_data_path, exist_ok=True)
        
    return forget_data, retain_data

@hydra.main(version_base=None, config_path="config", config_name="dusk")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    seed = cfg.seed
    set_random_seed(seed)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    config = AutoConfig.from_pretrained(model_id)

    # get the sequence of continual unlearning tasks
    task_list = os.getenv('TASK_LIST').split(',')
    task_list = [int(i) for i in task_list]
    # the order of unlearning tasks
    cfg.save_dir = os.path.join(cfg.save_dir, os.getenv('TASK_LIST').replace(',', '-'))
    # number of times to unlearn
    unlearn_times = task_list.index(cfg.task_id) + 1
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")

    if os.path.exists(os.path.join(curr_save_dir, 'eval_results-last', 'aggregate_stat.txt')):
        print(f'Task {cfg.task_id} already unlearned.')
        exit()

    if local_rank == 0:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    # get the unlearned model of the last unlearning task
    last_checkpoint_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times - 1}", "checkpoint-last")
    if (unlearn_times > 1) and (not os.path.exists(last_checkpoint_dir)):
        print('last checkpoint does not exist.')
        exit()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # process current forget set and retain set for unlearning
    curr_forget_data, curr_retain_data = get_task_data(curr_save_dir, tokenizer, cfg.num_formats)

    torch_format_dataset = ProfDataset(tokenizer=tokenizer,
                                               forget_data=curr_forget_data,
                                               retain_data=curr_retain_data,
                                               max_length=1024,)

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(
        torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)

    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (
                batch_size * gradient_accumulation_steps * num_devices)

    warmup_steps = steps_per_epoch if steps_per_epoch > 1 else 0

    if len(task_list) > 1:
        # only evaluate the last checkpoint of each task for continual unlearning by default
        save_steps = max_steps
    else:
        if cfg.save_steps == 'steps_per_epoch':
            save_steps = steps_per_epoch
        elif cfg.save_steps == 'last':
            save_steps = max_steps
        else:
            save_steps = cfg.save_steps

    if local_rank == 0:
        print("\n######### Unlearn Task %d #########" %
              (unlearn_times))
        print("Saving to: ", curr_save_dir)

    # load the config files for deepspeed
    if cfg.use_LoRA:
        ds_config = 'config/ds_config/lora.json'
    else:
        ds_config = 'config/ds_config/llama3.json'

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        output_dir=curr_save_dir,
        optim="paged_adamw_32bit",
        deepspeed=ds_config,
        save_steps=save_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=cfg.weight_decay,
        evaluation_strategy="no",
    )

    # for continual unlearning, load the target model from last task
    model_path = cfg.model_path if unlearn_times == 1 else last_checkpoint_dir
    # fix the reference model
    reference_model_path = cfg.model_path if cfg.fix_ref_model else model_path

    # load target LLM
    if cfg.use_LoRA and unlearn_times > 1:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
        )
        model.generation_config.do_sample = True
        if model_cfg["gradient_checkpointing"] == "true":
            model.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            target_modules=find_all_linear_names(model),
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            lora_dropout=cfg.LoRA.dropout,
        )
        model = PeftModel.from_pretrained(model, last_checkpoint_dir, config=peft_config, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
        )
        model.generation_config.do_sample = True
        if model_cfg["gradient_checkpointing"] == "true":
            model.gradient_checkpointing_enable()

        # Configure LoRA parameters
        if cfg.use_LoRA:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                target_modules=find_all_linear_names(model),
                r=cfg.LoRA.r,
                lora_alpha=cfg.LoRA.alpha,
                lora_dropout=cfg.LoRA.dropout,
            )
            model = get_peft_model(model, peft_config)

    # load reference model
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        config=config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
    )
    reference_model = reference_model.eval()

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        # the callback for computing metrics, None in this case since you're doing it in your callback
        compute_metrics=None,
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=custom_data_collator_forget,
        loss_type=cfg.forget_loss,
        ref_model=reference_model,
        beta=cfg.beta,
        forget_coeff=cfg.forget_coeff,
        regularization_coeff=cfg.regularization_coeff,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if cfg.forget_loss == 'RMU':
        # select the alyer and parameter to train
        layer_ids_to_train = [5, 6, 7]      
        param_ids_to_train = [6]

        for layer_idx, layer in enumerate(model.model.layers):
            for i, param in enumerate(layer.parameters()):
                if (layer_idx in layer_ids_to_train) and (i in param_ids_to_train):
                    param.requires_grad = True 
                else:
                    param.requires_grad = False

    print('Start Training ...')
    # Start training
    trainer.train()

    if local_rank == 0:
        if os.path.exists(os.path.join(curr_save_dir, f'checkpoint-{max_steps}')):
            if len(task_list) > 1 or cfg.save_steps == 'last':
                # continual
                shutil.move(os.path.join(curr_save_dir, f'checkpoint-{max_steps}'),
                            os.path.join(curr_save_dir, f'checkpoint-last'))
            else:
                # single
                if cfg.save_checkpoint:
                    shutil.copytree(os.path.join(curr_save_dir, f'checkpoint-{max_steps}'),
                                    os.path.join(curr_save_dir, f'checkpoint-last'))

        if os.path.exists(last_checkpoint_dir) and not cfg.save_checkpoint:
            # for evaluate last task
            # For continual unlearning, remove the last model checkpoint
            if os.path.exists(os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times - 1}", "eval_results-last")):
                shutil.rmtree(last_checkpoint_dir)
                print('Removed %s' % last_checkpoint_dir)


if __name__ == "__main__":
    main()