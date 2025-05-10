import subprocess
import os
import argparse

def evaluate_multiple_models(model_paths, tasks, output_dir, use_flash_attn=True):
    os.makedirs(output_dir, exist_ok=True)

    for model_path in model_paths:
        model_output_dir = os.path.join(output_dir, model_path.replace("/", "_")) 
        os.makedirs(model_output_dir, exist_ok=True)

        # Prepare model args
        model_args = f"pretrained={model_path},trust_remote_code=True"
        if use_flash_attn:
            model_args += ",attn_implementation=flash_attention_2"

        task_str = ",".join(tasks)

        command = [
            "accelerate", "launch", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", task_str,
            "--batch_size", "auto:8",
            "--output_path", model_output_dir,
            "--trust_remote_code"
        ]

        print(f"\n▶ Running lm-eval for: {model_path}")
        subprocess.run(command, check=True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True, help="List of model paths")
    parser.add_argument("--tasks", nargs="+", required=True, help="List of evaluation tasks")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    args = parser.parse_args()

    print(f"args.output_dir: {args.output_dir}")
    evaluate_multiple_models(
        model_paths=args.model_paths,
        tasks=args.tasks,
        output_dir=args.output_dir
    )
