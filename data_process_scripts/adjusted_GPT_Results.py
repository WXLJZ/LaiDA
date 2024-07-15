# --coding:utf-8--
import json

'''调整GPT生成后的数据的格式，使其满足微调所需的格式'''

with open("../preprocess_data/target_train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
index = 0
for entry in data:
    new_data.append({
        "metaphor_id": index,
        "input": entry["input"],
        "output": entry["prediction"]
    })
    index += 1

with open("../preprocess_data/train_train.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)
