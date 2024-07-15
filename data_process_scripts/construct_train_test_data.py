# --coding:utf-8--
import json
import random
from tqdm import tqdm

'''筛除已经处理好的部分训练集，余下的训练集用作微调的测试集使用'''

with open('../data/transfer_train.json', 'r', encoding='utf-8') as f:
    all_train_data = json.load(f)
with open('../Fine_tuning_data/Fine_tuning_train.json', 'r', encoding='utf-8') as f:
    train_train_data = json.load(f)

# 判断数据是否为已处理的数据
def judge(data):
    for entry in train_train_data:
        input_sentence = entry['input'].split('\n')[0]
        data_sentence = data['input'].split('\n')[0]
        if data_sentence == input_sentence:
            return True
    return False

def judge_type(data):
    right_option = data["input"].split('\n')[1]
    if '\t' in right_option and not judge(data):
        return 'multiple'
    elif '\t' not in right_option and not judge(data):
        return 'single'
    else:
        return 'have_processed'

# 统计multiple和single类别的数量
multiple_count = sum(1 for item in tqdm(all_train_data, desc="multiple detect") if judge_type(item) == 'multiple')
single_count = sum(1 for item in tqdm(all_train_data, desc="single detect") if judge_type(item) == 'single')
print(f"multiple_count: {multiple_count}, single_count: {single_count}")

# 计算需要筛选的multiple和single类别的数量
target_multiple_count = int(0.1 * 17000)
target_single_count = 17000 - target_multiple_count

# 使用random.sample进行随机抽样
multiple_samples = random.sample([item for item in tqdm(all_train_data, desc="random sample multiple") if judge_type(item) == 'multiple'], k=target_multiple_count)
single_samples = random.sample([item for item in tqdm(all_train_data, desc="random sample single") if judge_type(item) == 'single'], k=target_single_count)

# 合并并随机排序
selected_data = random.sample(multiple_samples + single_samples, k=17000)

train_test_data = []
index = 0
for data in tqdm(selected_data, desc="Constructing test data"):
    data["metaphor_id"] = index
    train_test_data.append(data)
    index += 1

with open('../preprocess_data/train_test_supplement.json', 'w', encoding='utf-8') as f:
    json.dump(train_test_data, f, ensure_ascii=False, indent=4)