# --coding:utf-8--
import argparse
from llamafactory import export_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and export model!")
    parser.add_argument("--model_name_or_path", type=str, help="The model name or path to the model.")
    parser.add_argument("--adapter_name_or_path", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--template", type=str, help="The type of model.")
    parser.add_argument("--finetuning_type", type=str, help="The type of fine-tuning.")
    parser.add_argument("--export_dir", type=str, help="The path you want to export the model.")
    parser.add_argument("--export_size", type=int, help="The file shard size (in GB) of the exported model.")
    parser.add_argument("--export_device", type=str, help="The device used in model export, use `auto` to accelerate exporting.")
    parser.add_argument("--export_legacy_format", action="store_true", help="Whether or not to save the `.bin` files instead of `.safetensors`.", default=True)
    args = parser.parse_args()
    export_model({
        "model_name_or_path": args.model_name_or_path,
        "adapter_name_or_path": args.adapter_name_or_path,
        "template": args.template,
        "finetuning_type": args.finetuning_type,
        "export_dir": args.export_dir,
        "export_size": args.export_size,
        "export_device": args.export_device,
        "export_legacy_format": args.export_legacy_format
    })
