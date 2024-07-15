import json
import torch
import random
import sys
import os

from Icl.templates import pretrained_templates,icl_templates, data_process_templates
from Icl.ex_retriver import Ex_Retriver
from tqdm import tqdm

def construct_instruct(json_path, save_path, retriever=None, selected_k=3, verbose=False):
    '''
    json_path: 原始，需要建立索引的数据路径
    save_path: 保存的路径
    '''
    with open(json_path, 'r', encoding='UTF-8') as fp:
        data = json.load(fp)
        target_data = []
        for d in tqdm(data, desc="Instruction Data Construction"):
            examples = retriever.search_examples(d['input'], selected_k=selected_k, verbose=verbose)

            examples_str = ""
            for id, example in enumerate(examples):
                examples_str += f'示例 {id+1}:\n输入："{example[0]}"\n输出："{example[1]}"\n'

            prompt = random.choice(icl_templates).format(example=examples_str, input=d['input'])

            if len(prompt) + len(d["output"]) > 1470:
                continue

            target_data.append({
                "metaphor_id": d["metaphor_id"],
                "instruction": prompt,
                "input": "",
                "output": d["output"]
            })

    # 直接将数据保存到对应位置
    with open(save_path, 'w', encoding='UTF-8') as fp:
        json.dump(target_data, fp, indent=4, ensure_ascii=False)

def construct_data_preprocess_instruct(json_path, save_path):
    with open(json_path, 'r', encoding='UTF-8') as fp:
        data = json.load(fp)
        for d in tqdm(data, desc="Data_Process Instruction Construction"):
            prompt = random.choice(data_process_templates).format(input=d['input'])
            d['instruction'] = prompt
            d['input'] = ""

    # 直接将数据保存到对应位置
    with open(save_path, 'w', encoding='UTF-8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)

def construct_pretrained_instruct(json_path, save_path, retriever=None, selected_k=3, verbose=False):
    with open(json_path, 'r', encoding='UTF-8') as fp:
        data = json.load(fp)
        for d in tqdm(data, desc="Pretrained Instruction Construction"):
            examples = retriever.search_examples(d['input'], selected_k=selected_k, verbose=verbose)

            examples_str = ""
            for id, example in enumerate(examples):
                examples_str += f'示例 {id + 1}:\n输入："{example[0]}"\n输出："{example[1]}"\n'

            prompt = random.choice(pretrained_templates).format(example=examples_str, input=d['input'])
            d['instruction'] = prompt
            d['input'] = ""

    # 直接将数据保存到对应位置
    with open(save_path, 'w', encoding='UTF-8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)

def pretrained_data_process(prefix, inst_prefix, method='random', selected_k=3):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if not os.path.exists(inst_prefix):
        os.makedirs(inst_prefix)

    train_path = prefix + 'train.json'
    inst_train_path = inst_prefix + method + '_inst_train.json'

    retriever = Ex_Retriver(ex_file=train_path, encode_method='random')
    construct_pretrained_instruct(train_path, inst_train_path, retriever, selected_k=selected_k)
    del retriever

    torch.cuda.empty_cache()

def train_data_process(is_process_data, prefix, inst_prefix, paths=None, method='random', selected_k=3):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if not os.path.exists(inst_prefix):
        os.makedirs(inst_prefix)
    if is_process_data:
        train_path = prefix + 'train.json'
        inst_train_path = inst_prefix + 'train_inst.json'
    else:
        train_path = prefix + 'Fine_tuning_train.json'
        inst_train_path = inst_prefix + method + '_inst_train.json'


    if method == 'random':
        if is_process_data:
            construct_data_preprocess_instruct(train_path, inst_train_path)
        else:
            retriever = Ex_Retriver(ex_file=train_path, encode_method='random')
            construct_instruct(train_path, inst_train_path, retriever, selected_k=selected_k)
            del retriever
    elif method == 'bert':
        retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='bert')
        construct_instruct(train_path, inst_train_path, retriever, selected_k=selected_k)
        del retriever
    elif method == 'gnn':
        retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='gnn')
        construct_instruct(train_path, inst_train_path, retriever, selected_k=selected_k)
        del retriever
    else:
        raise NotImplementedError

    torch.cuda.empty_cache()

def test_data_process(is_process_data, prefix, inst_prefix, paths=None, method='random', selected_k=3):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if not os.path.exists(inst_prefix):
        os.makedirs(inst_prefix)

    train_path = prefix + 'Fine_tuning_valid.json'
    if is_process_data:
        test_path = prefix + 'test.json'
        inst_test_path = inst_prefix + f'test_inst.json'
    else:
        test_path = prefix + 'Fine_tuning_test.json'
        inst_test_path = inst_prefix + method + '_inst_test.json'


    if method == 'random':
        if is_process_data:
            construct_data_preprocess_instruct(test_path, inst_test_path)
        else:
            retriever = Ex_Retriver(ex_file=train_path, encode_method='random')
            construct_instruct(test_path, inst_test_path, retriever, selected_k=selected_k)
            del retriever
    elif method == 'bert':
        retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='bert')
        construct_instruct(test_path, inst_test_path, retriever, selected_k=selected_k)
        del retriever
    elif method == 'gnn':
        retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='gnn')
        construct_instruct(test_path, inst_test_path, retriever, selected_k=selected_k)
        del retriever
    else:
        raise NotImplementedError

    torch.cuda.empty_cache()

    with open(inst_test_path, 'r', encoding='utf-8') as f:
        inst_test_data = json.load(f)

    return inst_test_data

def valid_data_process(prefix, inst_prefix, paths=None, method='random', selected_k=3):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if not os.path.exists(inst_prefix):
        os.makedirs(inst_prefix)

    train_path = prefix + 'Fine_tuning_valid.json'
    valid_path = prefix + 'Fine_tuning_valid.json'
    inst_valid_path = inst_prefix + method + '_inst_valid.json'

    if method == 'random':
        retriever = Ex_Retriver(ex_file=train_path, encode_method='random')
        construct_instruct(valid_path, inst_valid_path, retriever, selected_k=selected_k)
        del retriever
    elif method == 'bert':
        retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='bert')
        construct_instruct(valid_path, inst_valid_path, retriever, selected_k=selected_k)
        del retriever
    elif method == 'gnn':
        retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='gnn')
        construct_instruct(valid_path, inst_valid_path, retriever, selected_k=selected_k)
        del retriever
    else:
        raise NotImplementedError

    torch.cuda.empty_cache()

    with open(inst_valid_path, 'r', encoding='utf-8') as f:
        inst_valid_data = json.load(f)

    return inst_valid_data

