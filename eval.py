import json
import os
import warnings
import hydra
import torch

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import get_model_identifiers_from_yaml
from metrics.rouge_eval import rouge_forget_score
from metrics.qa_eval import qa_general_eval_score, qa_specific_eval_score
from metrics.mia_eval import mia_eval_score

warnings.filterwarnings('ignore')

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    task_list = os.getenv('TASK_LIST').split(',')
    task_list = [int(i) for i in task_list]
    cfg.save_dir = os.path.join(cfg.save_dir, os.getenv('TASK_LIST').replace(',', '-'))

    unlearn_times = task_list.index(cfg.task_id) + 1
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")

    if cfg.forget_loss == 'TV':
        curr_checkpoint_dir = os.path.join(curr_save_dir, f"checkpoint-{cfg.eval_unlearn_step}-tv")
    else:
        curr_checkpoint_dir = os.path.join(curr_save_dir, f"checkpoint-{cfg.eval_unlearn_step}")
        
    if cfg.eval_unlearn_step == 0:
        curr_checkpoint_dir = cfg.model_path
    else:
        if not os.path.exists(curr_checkpoint_dir):
            print(f'{curr_checkpoint_dir} does not exist.')
            exit()

    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')
    if os.path.exists(os.path.join(curr_eval_dir, 'aggregate_stat.csv')):
        print(f'{curr_eval_dir} already evaluated.')
        exit()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)

    if cfg.use_LoRA:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        model = PeftModel.from_pretrained(model, curr_checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            curr_checkpoint_dir,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
    model = model.eval()

    # Evaluation of ROUGE score
    rouge_forget_score(cfg, unlearn_times, model, tokenizer)
    
    # Evaluation of General QA
    qa_general_eval_score(cfg, unlearn_times, model, tokenizer) 
    
    # Evaluation of Specific QA
    qa_specific_eval_score(cfg, unlearn_times, model, tokenizer) 

    # MIA Evaluation
    _, _, _, auc = mia_eval_score(cfg, unlearn_times, model, tokenizer)
    out = {}
    if "target" in cfg.model_path:
        retrain_path = "./results/llama3-8b/NONE+GD/seed_1001/epoch1_1e-05_taskformats_5/1/unlearn_times_1/eval_results-0/mia/MIA_retrain_AUROC.json"

        with open(retrain_path, "r") as f:
            AUC_RETRAIN = json.load(f)
    
        ## PrivLeak (from MUSE)
        out['privleak'] = (auc["forget_holdout_mink++_0.4"] - AUC_RETRAIN["forget_holdout_mink++_0.4"]) / AUC_RETRAIN["forget_holdout_mink++_0.4"] * 100
        ## RetainDeviation
        out['RetainDeviation'] = (abs((auc["holdout_retainmink++_0.4"]-AUC_RETRAIN["holdout_retainmink++_0.4"]) / AUC_RETRAIN["holdout_retainmink++_0.4"])) * 100
        
        mia_path = os.path.join(curr_eval_dir, f"mia/privleak_RD_target.json")
    else:
        # for the case of retrain
        out['privleak'] = 0
        out['RetainDeviation'] = 0
        mia_path = os.path.join(curr_eval_dir, f"mia/privleak_RD_retrain.json")

    with open(mia_path, 'w') as f:
        json.dump(out, f, indent=4)

    # Do not save checkpoint at last step for memory efficiency
    #if unlearn_times == len(task_list) and not cfg.save_checkpoint:
    #    if (os.path.exists(curr_checkpoint_dir)) and (cfg.eval_unlearn_step != 0):
    #        shutil.rmtree(curr_checkpoint_dir)


if __name__ == "__main__":
    main()