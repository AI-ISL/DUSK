import os
from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
from scipy.stats import bootstrap
import numpy as np

from tqdm.contrib import tzip
from typing import List, Dict, Union
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

def read_json(fpath: str) -> Union[List, Dict]:
    data = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line)) 
    return data


def read_json_partial(fpath: str, num: int) -> List[dict]:
    """Reads a portion of a JSONL file.
    
    Args:
        fpath (str): Path to the JSONL file
        num (int): An integer from 2 to 5 (1=20 lines, 2=40 lines, 3=60 lines, 4=80 lines)

    Returns:
        List[dict]: A list of parsed JSON objects
    """
    if num not in {2, 3, 4, 5}:
        raise ValueError("num must be an integer between 2 and 5.")
    
    limit = (num - 1) * 20 
    data = []
    
    with open(fpath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data.append(json.loads(line))
    
    return data

def write_json(obj: Dict | List, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return json.dump(obj, f)
    
def qa_general_eval_score(cfg, unlearn_times, model, tokenizer): 

    data_icl = read_json(f"data/Prof/eval/icl.jsonl")
    data = read_json(f"data/Prof/eval/GeneralQA.jsonl")
    
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')
    os.makedirs(curr_eval_dir, exist_ok=True)
    
    agg, log = eval(
            questions=[d['question'] for d in data],
            answers=[d['answer'] for d in data],
            icl_qs=[d['question'] for d in data_icl],
            icl_as=[d['answer'] for d in data_icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=32
        )
    write_json(agg, os.path.join(curr_eval_dir, "qa/qa_general_agg.json"))
    write_json(log, os.path.join(curr_eval_dir, "qa/qa_general_log.json"))

    
def qa_specific_eval_score(cfg, unlearn_times, model, tokenizer):
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')
    os.makedirs(curr_eval_dir, exist_ok=True)

    data_icl = read_json(f"data/Prof/eval/icl.jsonl")
    data_forget = read_json(f"data/Prof/eval/SpecificForgetQA.jsonl")

    agg_forget, log_forget = eval(
            questions=[d['question'] for d in data_forget],
            answers=[d['answer'] for d in data_forget],
            icl_qs=[d['question'] for d in data_icl],
            icl_as=[d['answer'] for d in data_icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=32
        )
    write_json(agg_forget, os.path.join(curr_eval_dir, "qa/qa_specific_forget_agg.json"))
    write_json(log_forget, os.path.join(curr_eval_dir, "qa/qa_specific_forget_log.json"))
    
    data_retain = read_json(f"data/Prof/eval/SpecificRetainQA.jsonl")    
    agg_retain, log_retain = eval(
            questions=[d['question'] for d in data_retain],
            answers=[d['answer'] for d in data_retain],
            icl_qs=[d['question'] for d in data_icl],
            icl_as=[d['answer'] for d in data_icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=32
        )
    write_json(agg_retain, os.path.join(curr_eval_dir, "qa/qa_specific_retain_agg.json"))
    write_json(log_retain, os.path.join(curr_eval_dir, "qa/qa_specific_retain_log.json"))

    


def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def eval(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32
):
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = RougeEvalLogger()
    general_prompt: str = ""

    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"

    for question, answer in tzip(questions, answers):
        prompt = general_prompt + f"Question: {question}\nAnswer: "

        # Encode the `prompt` into `input_ids`
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True).input_ids

        # Use the `model` to generate the continuation of the `input_ids`.
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

        output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        logger.log(prompt, answer, output, question=question)

    return logger.report()