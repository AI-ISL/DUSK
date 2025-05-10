import json
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path
from typing import List
import torch.nn.functional as F


def read_text(fpath: str) -> str:
    with open(fpath, 'r') as f:
        return f.read()

def dataset_to_json(dataset, filename, ):
    data_nums = len(dataset)
    with open(filename, "w") as f:
        for i in range(data_nums):
            row_data = dataset[i]
            json_data = json.dumps(row_data)
            f.write(json_data)
            f.write('\n')

# adopt from TOFU: https://github.com/locuslab/tofu/blob/80159d8ea39edf147fb09cd82aefa08e506e6718/data_module.py#L8
def convert_raw_forget_data_to_model_format(tokenizer, max_length, question, answer, model_configs, mask=True):
    question_start_token, question_end_token, answer_token = model_configs[
        'question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']

    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer

    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if mask:
        if len(encoded.input_ids) == max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
        # change label to -100 for question tokens
        for i in range(num_question_tokens): label[i] = -100
    else:
        label = pad_input_ids

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs[
        'question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)

    # change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)

class DefaultDataset(Dataset):

    def __init__(
        self,
        file_path: str | None = None,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 1024,
        add_bos_token: bool = True,
        input_ids: List[torch.Tensor] | None = None
    ):
        self.input_ids = []
        
        if file_path:
            assert Path(file_path).suffix == '.txt'
            tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
            assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

            if add_bos_token:
                self.input_ids = [
                    F.pad(
                        tokens[i : i + max_len - 1], (1, 0),
                        value=tokenizer.bos_token_id
                    )
                    for i in range(0, len(tokens), max_len - 1)
                ]
            else:
                self.input_ids = [
                    tokens[i : i + max_len]
                    for i in range(0, len(tokens), max_len)
                ]

            if len(self.input_ids[-1]) < max_len:
                self.input_ids[-1] = torch.concat(
                    [self.input_ids[-1], self.input_ids[0]], dim=-1
                )[:max_len]
        
        if input_ids:
            self.input_ids.extend(input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]

    def __len__(self):
        return len(self.input_ids)

    def __add__(self, other):
        assert isinstance(other, DefaultDataset), "Can only merge with another DefaultDataset"
        return DefaultDataset(input_ids=self.input_ids + other.input_ids)

    def get_collate_fn(self):
        def collate_fn(batch: List[torch.Tensor]):
            batch = torch.stack(batch)
            return {
                "input_ids": batch,
                "labels": batch.clone()
            }
        return collate_fn

class ProfDataset(Dataset):
    def __init__(self, tokenizer, forget_data, retain_data, max_length=1024):
        super(ProfDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = forget_data
        self.retain_data = retain_data

        print("forget_data: ", len(self.forget_data))
        print("retain_data: ", len(self.retain_data))

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        # idx must be random (Unlearn)
        retain_idx = torch.randint(0, len(self.retain_data), (1,)).item()
        return [(self.forget_data[idx], self.forget_data[idx].clone(), torch.ones_like(self.forget_data[idx])), 
                (self.retain_data[retain_idx], self.retain_data[retain_idx].clone(), torch.ones_like(self.retain_data[retain_idx]))]

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(
        attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks

def custom_data_collator_forget(samples):
    rets = []

    # Extracting samples for each data type
    data_types = ["forget", "retain"]
    samples_dict = {data_type: [sample[i] for sample in samples] for i, data_type in enumerate(data_types)}

    for data_type in data_types:
        data = samples_dict[data_type]

        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]

        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))

    return rets


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss
