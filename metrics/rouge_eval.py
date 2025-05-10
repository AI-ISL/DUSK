import os
from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
from scipy.stats import bootstrap
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein


from tqdm.contrib import tzip
from typing import List
import json
from typing import List, Dict

class RougeEvalLogger:

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=False
        )
        self.history = []


    def log(self, prompt: str, gt: str, output: str, question: str | None = None):
        score = self.scorer.score(gt, output)
        d = {
            'prompt': prompt,
            'gt': gt,
            'response': output,
            'rougeL': score['rougeL'].fmeasure,
            'rougeL_recall': score['rougeL'].recall,
            'rouge1': score['rouge1'].fmeasure,
            'rouge1_recall': score['rouge1'].recall
        }
        if question is not None: d['question'] = question
        self.history.append(d)


    def report(self) -> Tuple[Dict, Dict]:
        agg = {}
        for key, val in self.history[0].items():
            if isinstance(val, str): continue
            vals: List[float] = [item[key] for item in self.history]
            agg[f"max_{key}"] = max(vals)
            agg[f"mean_{key}"] = sum(vals) / len(vals)
            agg[f"{key}_ci_lo"], agg[f"{key}_ci_hi"] = bootstrap(
                (vals,), np.mean,
                confidence_level=0.95,
                method='percentile'
            ).confidence_interval
        return agg, self.history

def read_json(fpath: str) -> Dict | List:
    with open(fpath, 'r') as f:
        return json.load(f)

def write_json(obj: Dict | List, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return json.dump(obj, f)
    
def trans(input_file, tokenizer, prompt_size, gt_size):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    tokens = tokenizer.encode(text, add_special_tokens=True)

    chunk_size = prompt_size + gt_size
    data = []

    for i in range(0, len(tokens) - chunk_size + 1, 5):
        chunk = tokens[i : i + chunk_size]

        prompt_tokens = chunk[:prompt_size]
        gt_tokens = chunk[prompt_size: prompt_size + gt_size]  

        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        gt_text = tokenizer.decode(gt_tokens, skip_special_tokens=True)

        data.append({"prompt": prompt_text, "gt": gt_text})
    
    return data


def rouge_forget_score(cfg, unlearn_times, model, tokenizer):
    gt_size = 128

    data = read_json(f"data/Prof/eval/verbatim.json")    

    agg, log = eval_rouge(
                        prompts=[d['prompt'] for d in data],
                        gts=[d['gt'] for d in data],
                        model=model, tokenizer=tokenizer,
                        max_new_tokens=gt_size,
                        save_dir=cfg.save_dir,
                        unlearn_times=unlearn_times,
                        eval_unlearn_step=cfg.eval_unlearn_step
                    )

    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')
    os.makedirs(curr_eval_dir, exist_ok=True)
    write_json(agg, os.path.join(curr_eval_dir, f"rouge/rouge_agg.json"))
    write_json(log, os.path.join(curr_eval_dir, f"rouge/rouge_log.json"))

    agg2, log2 = eval_cosine(
            prompts=[d['prompt'] for d in data],
            gts=[d['gt'] for d in data],
            model=model, tokenizer=tokenizer,
            max_new_tokens=gt_size,
            save_dir=cfg.save_dir,
            unlearn_times=unlearn_times,
            eval_unlearn_step=cfg.eval_unlearn_step
        )   
    
    agg3, log3 = eval_lcs(
            prompts=[d['prompt'] for d in data],
            gts=[d['gt'] for d in data],
            model=model, tokenizer=tokenizer,
            max_new_tokens=gt_size,
            save_dir=cfg.save_dir,
            unlearn_times=unlearn_times,
            eval_unlearn_step=cfg.eval_unlearn_step
        )   
    
    agg4, log4 = eval_levenshtein(
            prompts=[d['prompt'] for d in data],
            gts=[d['gt'] for d in data],
            model=model, tokenizer=tokenizer,
            max_new_tokens=gt_size,
            save_dir=cfg.save_dir,
            unlearn_times=unlearn_times,
            eval_unlearn_step=cfg.eval_unlearn_step
        )
    
    write_json(agg2, os.path.join(curr_eval_dir, "rouge/cosine_agg.json"))
    write_json(log2, os.path.join(curr_eval_dir, "rouge/cosine_log.json"))
    write_json(agg3, os.path.join(curr_eval_dir, "rouge/lcs_agg.json"))
    write_json(log3, os.path.join(curr_eval_dir, "rouge/lcs_log.json"))
    write_json(agg4, os.path.join(curr_eval_dir, "rouge/levenshtein_agg.json"))
    write_json(log4, os.path.join(curr_eval_dir, "rouge/levenshtein_log.json"))


def eval_rouge(
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens : int = 128,
    save_dir: str = None,
    unlearn_times: int = 0,
    eval_unlearn_step: int = 0
):
    logger = RougeEvalLogger()
    for prompt, gt in tzip(prompts, gts):
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True
        ).input_ids
        assert len(input_ids) == 1

        gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]

        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        output_ids = output_ids[:, len(input_ids[0]):]
        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        gt_short = tokenizer.batch_decode(
            gt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        logger.log(prompt, gt_short, output)

        try:
            agg, log = logger.report()
            curr_save_dir = os.path.join(save_dir, f"unlearn_times_{unlearn_times}")
            curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{eval_unlearn_step}')
            os.makedirs(curr_eval_dir, exist_ok=True)
    
            write_json(agg, os.path.join(curr_eval_dir, "rouge/rouge_agg.json"))
            write_json(log, os.path.join(curr_eval_dir, "rouge/rouge_log.json"))
        except:
            print("Error")


    return logger.report()


def eval_cosine(
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens : int = 128,
    save_dir: str = None,
    unlearn_times: int = 0,
    eval_unlearn_step: int = 0
):
    cmodel = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=torch.device('cuda'))
    scores = []
    logs = []
    for prompt, gt in tzip(prompts, gts):
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True
        ).input_ids
        assert len(input_ids) == 1

        gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]

        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        output_ids = output_ids[:, len(input_ids[0]):]
        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        gt_short = tokenizer.batch_decode(
            gt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]

        gen_embedding = cmodel.encode(output, show_progress_bar=False)
        gt_embedding = cmodel.encode(gt_short, show_progress_bar=False)
        cosine_sim = cosine_similarity([gen_embedding], [gt_embedding])[0][0]
        cosine_sim = float(max(0, cosine_sim))
        scores.append(cosine_sim)
        logs.append({"prompt": prompt, "gt": gt_short, "output": output, "cosine_similarity": cosine_sim})
    agg = {"mean_cosine_similarity": sum(scores) / len(scores) if scores else 0}

    
    return agg, logs

def eval_lcs( ##word-level lcs
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens: int = 128,
    save_dir: str = None,
    unlearn_times: int = 0,
    eval_unlearn_step: int = 0
):
    scores = []
    logs = []
    for prompt, gt in tzip(prompts, gts):
        input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=True).input_ids
        assert len(input_ids) == 1

        gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]

        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        output_ids = output_ids[:, len(input_ids[0]):]
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        gt_short = tokenizer.batch_decode(gt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        lcs_score = word_level_lcs(gt_short, output)
        scores.append(lcs_score)
        logs.append({"prompt": prompt, "gt": gt_short, "output": output, "lcs_score": lcs_score})

    agg = {"mean_lcs_score": sum(scores) / len(scores) if scores else 0}

    return agg, logs

def word_level_lcs(seq1: str, seq2: str) -> float:
    """Word-level LCS"""
    words1 = seq1.split()
    words2 = seq2.split()
    m, n = len(words1), len(words2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    max_length = max(m, n) if max(m, n) > 0 else 1 

    return lcs_length

def eval_levenshtein(   ##character-level 
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens: int = 128,
    save_dir: str = None,
    unlearn_times: int = 0,
    eval_unlearn_step: int = 0
):
    scores = []
    logs = []
    for prompt, gt in tzip(prompts, gts):
        input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=True).input_ids
        assert len(input_ids) == 1

        gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]

        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        output_ids = output_ids[:, len(input_ids[0]):]
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        gt_short = tokenizer.batch_decode(gt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Levenshtein Distance
        lev_dist = Levenshtein.distance(gt_short, output)
        max_length = max(len(gt_short), len(output)) if max(len(gt_short), len(output)) > 0 else 1
        lev_sim = 1 - (lev_dist / max_length) 

        # scores.append(lev_sim)
        scores.append(lev_dist)
        logs.append({"prompt": prompt, "gt": gt_short, "output": output, "levenshtein_similarity": lev_sim})

    agg = {"mean_levenshtein_similarity": sum(scores) / len(scores) if scores else 0}

    return agg, logs