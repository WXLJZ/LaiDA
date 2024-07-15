from tqdm import tqdm

from llamafactory import ChatModel


class Model:
    def __init__(self, model_name_or_path, checkpoint_dir, temperature=0.1, top_p=0.9, finetuning_type="lora"):
        args = {
            "model_name_or_path": model_name_or_path,
            "adapter_name_or_path": checkpoint_dir,
            "template": "qwen",
            "finetuning_type": finetuning_type,
            "temperature": temperature,
            "top_p": top_p,
        }  
        self.chat_model = ChatModel(args=args)

    def generate(self, query):
        res = self.chat_model.chat(query)
        return res


