import time
import pandas as pd
import re
import random
import json
import os
import torch

from tqdm import tqdm
from Icl.predict import Model

def parser_output(output):
    output_list = output.split(": ")
    # 检查是否成功分割并且存在至少两个部分
    if len(output_list) >= 2:
        option_letter = output_list[0].strip()
    else:
        raise ValueError("The output string is not in the correct format")
    return option_letter

def parser_predict_output(output):
    matches = re.findall(r'[ABCD]: \[[^\]]*\]', output)
    if matches:
        option = matches[0]
        return option
    else:
        raise ValueError("The output string is not in the correct format")


def get_accuracy(y_trues, y_preds):
    correct = 0
    for true, pred in zip(y_trues, y_preds):
        if true == pred:
            correct += 1
    return correct / len(y_trues)

def evaluate_test(test_data, model_name_or_path, checkpoint_dir, is_process_data, temperature=0.1, top_p=0.9, finetuning_type="lora", results_path_prefix=None, method=None):
    model = Model(model_name_or_path, checkpoint_dir, temperature, top_p, finetuning_type)

    y_trues, y_preds = [], []
    all_results = []  # 用于保存全部记录
    error_records = []  # 用于保存错误的记录
    error_id = 0  # 记录错误记录的条数
    for data in tqdm(test_data, desc="Predicting the Test_Dataset"):
        instruction = data['instruction']

        message = [
            {"role": "user", "content": instruction}
        ]

        pred = model.generate(message)[0].response_text
        true = data['output']
        if is_process_data:
            all_results.append({
                'metaphor_id': data['metaphor_id'],
                'instruction': instruction,
                'output': pred
            })
            if not os.path.exists(results_path_prefix):
                os.makedirs(results_path_prefix)
            if len(all_results) % 2000 == 0:
                number = len(all_results) // 2000
                results_path = f"{results_path_prefix}temp_results/Fixed_ICL_processed_data_results_{number}.json"
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
        else:
            try:
                pred = parser_predict_output(pred)
                pred_option = parser_output(pred)
                true_option = parser_output(true)
            except Exception as e:
                # 如果在生成或解析预测过程中出现异常，将预测结果设置为异常信息，将预测结果设置为空列表
                pred = str(e)
                pred_option = []

            y_trues.append(true_option)
            y_preds.append(pred_option)

            all_results.append({
                'metaphor_id': data['metaphor_id'],
                'instruction': instruction,
                'true': true,
                'output': pred
            })
            # 如果预测的输出与真实的输出不匹配，将它们添加到错误记录中
            if pred != true:
                error_id += 1
                error_records.append({
                    'error_id': str(error_id),
                    'metaphor_id': data['metaphor_id'],
                    'instruction': instruction,
                    'true': true,
                    'pred': pred
                })

    if not os.path.exists(results_path_prefix):
        os.makedirs(results_path_prefix)

    if is_process_data:
        results_path = f"{results_path_prefix}Fixed_ICL_processed_data_results_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
    else:
        # 保存错误记录到json文件
        if error_records:
            error_path = f"{results_path_prefix}{method}_test_set_error_records_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
            results_path = f"{results_path_prefix}{method}_test_set_all_results_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_records, f, ensure_ascii=False, indent=4)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
    # 计算准确度
    accuracy = get_accuracy(y_trues, y_preds)

    torch.cuda.empty_cache()

    return accuracy

def evaluate_valid(valid_data, model_name_or_path, checkpoint_dir, is_process_data, temperature=0.1, top_p=0.9, finetuning_type="lora", results_path_prefix=None, method=None):
    model = Model(model_name_or_path, checkpoint_dir, temperature, top_p, finetuning_type)

    y_trues, y_preds = [], []
    all_results = [] # 用于保存全部记录
    error_records = []  # 用于保存错误的记录
    error_id = 0 # 记录错误记录的条数
    # for data in tqdm(random.sample(test_data, 100)):
    for data in tqdm(valid_data, desc="Predicting the Validation_Dataset"):
        instruction = data['instruction']

        message = [
            {"role": "user", "content": instruction}
        ]

        pred = model.generate(message)[0].response_text
        true = data['output']
        if is_process_data:
            all_results.append({
                'metaphor_id': data['metaphor_id'],
                'instruction': instruction,
                'context': data['context'],
                'output': pred
            })
        else:
            try:
                pred = parser_predict_output(pred)
                pred_option = parser_output(pred)
                true_option = parser_output(true)
            except Exception as e:
                # 如果在生成或解析预测过程中出现异常，将预测结果设置为异常信息，将预测结果设置为空列表
                pred = str(e)
                pred_option = []

            y_trues.append(true_option)
            y_preds.append(pred_option)

            all_results.append({
                'metaphor_id': data['metaphor_id'],
                'instruction': instruction,
                'true': true,
                'output': pred
            })
            # 如果预测的输出与真实的输出不匹配，将它们添加到错误记录中
            if pred != true:
                error_id += 1
                error_records.append({
                    'error_id': str(error_id),
                    'metaphor_id': data['metaphor_id'],
                    'instruction': instruction,
                    'true': true,
                    'predict': pred
                })

    if not os.path.exists(results_path_prefix):
        os.makedirs(results_path_prefix)

    if is_process_data:
        results_path = f"{results_path_prefix}Fixed_ICL_processed_data_results_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
    else:
        # 保存错误记录到json文件
        if error_records:
            error_path = f"{results_path_prefix}{method}_validation_set_error_records_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
            results_path = f"{results_path_prefix}{method}_validation_set_all_results_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_records, f, ensure_ascii=False, indent=4)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

    # 计算准确度
    accuracy = get_accuracy(y_trues, y_preds)

    torch.cuda.empty_cache()

    return accuracy
