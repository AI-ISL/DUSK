import os
import torch
import warnings
import hydra
import torch

from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_model_identifiers_from_yaml, set_random_seed
from transformers import AutoModelForCausalLM


warnings.filterwarnings('ignore')

@hydra.main(version_base=None, config_path="config", config_name="dusk")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    seed = cfg.seed
    set_random_seed(seed)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

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
    if local_rank == 0:
        if cfg.forget_loss == 'TV':
            reinforce_checkpoint_dir = os.path.join(curr_save_dir, "checkpoint-last")
            if not os.path.exists(reinforce_checkpoint_dir):
                print("Error: Reinforce model checkpoint not found:", reinforce_checkpoint_dir)
                exit(1)
                
            tv_output_dir = os.path.join(curr_save_dir, "checkpoint-last-tv")
            reference_model_dir = cfg.model_path

            unlearn(
                model_dir=reference_model_dir,
                out_dir=tv_output_dir,
                some_pt_model_dir=reference_model_dir,       
                some_ft_model_dir=reinforce_checkpoint_dir,  
                alpha=1.0
            )
            
            print(f"TV unlearned model saved to {tv_output_dir}")

def load_model(model_dir: str, state_dict=None, **kwargs) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        **kwargs
    )
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    return model

def unlearn(
    model_dir: str,
    out_dir: str | None = None,
    some_pt_model_dir: str | None = None,
    some_ft_model_dir: str | None = None,
    alpha: float = 1.0,
    ):
    if some_pt_model_dir is None or some_ft_model_dir is None:
        raise ValueError("Task vector (ilharco2023) requires some pretrained & finetuned models!")

    task_vector = TaskVector(
        pretrained_state_dict = load_model(some_pt_model_dir).state_dict(),
        finetuned_state_dict = load_model(some_ft_model_dir).state_dict()
    )
    
    if not task_vector.is_nonzero():
        raise ValueError("Zero task vector encountered!")

    neg_task_vector = -task_vector
    model = load_model(model_dir)
    new_state_dict = neg_task_vector.apply_to(pretrained_model=model, scaling_coef=alpha, in_place=False)

    new_model = load_model(model_dir, device_map='auto')
    new_model.load_state_dict(new_state_dict, strict=False)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        new_model.save_pretrained(out_dir)
    return new_model


class TaskVector():
    def __init__(self,
                 pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None,
                 pretrained_state_dict=None, finetuned_state_dict=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert (
                (pretrained_checkpoint is not None and finetuned_checkpoint is not None)
                or
                (pretrained_state_dict is not None and finetuned_state_dict is not None)
            )
            with torch.no_grad():
                if pretrained_state_dict is None:
                    pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                if finetuned_state_dict is None:
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def is_nonzero(self):
        return any([(self.vector[key] != 0).any() for key in self.vector])

    def apply_to(self, pretrained_model, scaling_coef=1.0, in_place=False):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        if in_place:
            pretrained_model.load_state_dict(new_state_dict, strict=False)
        return new_state_dict
    
if __name__ == "__main__":
    main()