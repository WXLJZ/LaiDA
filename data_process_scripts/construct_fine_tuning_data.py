# --coding:utf-8--
import json
import re
import random

def parser_data(data):
    '''解析数据'''
    pattern = r'\[[^\[\]]*\]'
    matches = re.findall(pattern, data)
    return matches

'''构造最终的微调数据'''
def construct_fine_tuning_data(data_path, save_path):
    # 读取数据
    origin_data = json.load(open(data_path, 'r', encoding='utf-8'))
    target_data = []
    index = 0 # 对数据重新编号
    for data_entry in origin_data:
        if len(parser_data(data_entry['predict'])) < 3:
            continue
            # raise ValueError(f"Metaphor {data_entry['metaphor_id']} , The number of error options is less than 3")
        else:
            error_options_list = parser_data(data_entry['predict'])[:3]
        context_list = data_entry['context'].split('\n')
        input_sentence = context_list[0]
        right_answer = context_list[1]
        options_list = [right_answer] + error_options_list
        # 随机打乱选项列表
        random.shuffle(options_list)
        # 分配选项到A、B、C、D（选项列表长度固定为4）
        A, B, C, D = options_list
        A = 'A: ' + A
        B = 'B: ' + B
        C = 'C: ' + C
        D = 'D: ' + D
        # 找出正确答案的索引并转换为对应的字母（A、B、C、D）
        answer_index = options_list.index(right_answer)
        answer_letter = chr(ord('A') + answer_index)
        answer = answer_letter + ': ' + right_answer
        # 构造输入文本
        input_text = input_sentence + '\n' + A + '\n' + B + '\n' + C + '\n' + D + '\n'
        target_data.append({
            'metaphor_id': index,
            # 'input_sentence': input_sentence,
            # 'right_answer': right_answer,
            # 'error_options_list': error_options_list,
            'input': input_text,
            'output': answer
        })
        index += 1
    # 保存数据
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    data_path = '../preprocess_data/results/Fixed_ICL_processed_data_results_2024-06-19-22-59-32.json'
    save_path = '../Fine_tuning_data/Fine_tuning_train.json'
    construct_fine_tuning_data(data_path, save_path)

