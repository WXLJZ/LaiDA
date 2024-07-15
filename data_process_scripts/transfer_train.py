# --coding:utf-8--
import json
from collections import defaultdict

'''将训练数据转换为待处理(生成干扰选项)的数据'''

# 读取训练数据
with open("../data/original_train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 合并后的数据字典(初始化)
merged_data = defaultdict(lambda: {
    "metaphor_id": None,
    "context": "",
    "tenor": [],
    "vehicle": [],
    "ground": [],
    "sub_id": 0
})

# 合并相同 metaphor_id 的数据
for entry in data:
    mid = entry["metaphor_id"]
    merged_data[mid]["metaphor_id"] = mid
    merged_data[mid]["context"] = entry["context"]
    if entry["tenor"] not in merged_data[mid]["tenor"]:
        merged_data[mid]["tenor"].append(entry["tenor"])
    merged_data[mid]["vehicle"].append(entry["vehicle"])
    merged_data[mid]["ground"].append(entry["ground"])

transfer_train_data = []
for key, value in merged_data.items():
    context = value["context"].replace("\n", " ")
    if len(value["tenor"]) == 1:
        tenor = value["tenor"][0]
    else:
        tenor = "\t".join(f"{i + 1}. {t}" for i, t in enumerate(value["tenor"]))
    if len(value["vehicle"]) == 1:
        vehicle = value["vehicle"][0]
    else:
        vehicle = "\t".join(f"{i + 1}. {v}" for i, v in enumerate(value["vehicle"]))
    if len(value["ground"]) == 1:
        ground = value["ground"][0]
    else:
        ground = "\t".join(f"{i + 1}. {g}" for i, g in enumerate(value["ground"]))
    metaphor_components_tuple = "[" + tenor + ", " + vehicle + ", " + ground + "]"
    input_text = context + "\n" + metaphor_components_tuple
    transfer_train_data.append({
        "metaphor_id": value["metaphor_id"],
        "input": input_text,
        "output": ""
    })

with open("../data/train.json", "w", encoding="utf-8") as f:
    json.dump(transfer_train_data, f, ensure_ascii=False, indent=4)