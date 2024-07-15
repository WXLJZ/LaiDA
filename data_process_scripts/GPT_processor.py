from openai import OpenAI
import openai
import time
import json
from tqdm import tqdm
import argparse

'''生成干扰选项'''

MAX_LENGTH = 2048
# Modify the api key with yours
client = OpenAI(api_key="sk-WuOT5Rhslw6RpNUZA8Ef2aC96b7943A386B619891048C1F4", base_url="https://www.jcapikey.com/v1")

parser = argparse.ArgumentParser(description='Generate some error options')

parser.add_argument('--train_data_file', type=str, help='Train data file path', default="../data/transfer_train.json")
parser.add_argument('--target_data_file', type=str, help='Processed file path', default="../preprocessed_data/target_train_data.json")

args = parser.parse_args()

train_data_file = args.train_data_file
target_data_file = args.target_data_file

def get_answer(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=1024,
        temperature=0.2,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=messages
    )

    return response

with open(train_data_file, encoding="utf-8") as f:
    data = json.load(f)
    data = data[:3000] # 处理前3000个数据
    new_data = []
    index = 1
    for entry in tqdm(data):
        input = entry["input"].strip()

        prompt = (f"我需要你帮助我完成一项复杂任务，任务描述为：给定一个隐喻句和包含隐喻成分的正确三元组（本体, 喻体, 描述本体和喻体之间关系的共性）。请你根据给定的隐喻句生成三个与给定三元组类似但不同的错误隐喻成分三元组。另外，请注意可能存在多组隐喻成分。以下示例供你参考：\n"
                  f"示例1：\n"
                  f"输入：每一片风景都承载着千年的文化，就像是大自然为我们写下的诗篇。\n[自然风景, 诗篇, 承载着文化]"
                  f"输出：[诗文, 自然景观, 承载着文化]\n[诗文, 人工景观, 承载着文化]\n[人工景观, 诗篇, 承载着文化]"
                  f"示例2：\n"
                  f"输入：高老头脸上的表情只有一个譬喻可以形容，仿佛一口锅炉贮满了足以翻江倒海的水汽，一眨眼之间被一滴冷水化得无影无踪\n[高老头脸上的表情, 1. 锅炉中翻腾的汽水\t2. 锅炉中浇了冷水, 1. 激动\t2. 平静]"
                  f"输出：[1. 锅炉中翻腾的汽水\t2. 锅炉中浇了冷水, 高老头脸上的表情, 激动又平静]\n[高老头的脸, 1. 锅炉中翻腾的汽水\t2. 锅炉中浇了冷水, 激动又平静]\n[高老头的脸, 1. 锅炉中翻腾的汽水\t2. 锅炉中浇了冷水, 激动又愤怒]"
                  f"注意输出格式与示例保持一致！现在，请你处理以下的输入吧！输入：\n{input}\n输出：\n")
        # print(prompt)
        messages = [
            {
                "role": "system",
                "content": "你是一位文学领域的专家！"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        try:
            response = get_answer(messages)
            new_datum = {
                'metaphor_id': entry["metaphor_id"],
                'input': input
            }
            prediction = response.choices[0].message.content
            new_datum["prediction"] = prediction
            new_data.append(new_datum)
            if len(new_data) % 100 == 0:
                json.dump(new_data,
                          open(f"../preprocess_data/temp_data/temp_data_{index + 1}.json", 'w', encoding="utf-8"),
                          indent=4, ensure_ascii=False)
                index += 1
        except openai.RateLimitError as e:
            print("遇到 RateLimitError，5s后尝试重连OpenAI...")
            # 等待一段时间，避免过于频繁地重试
            time.sleep(5)
            # 重新预测，继续捕获异常，看看会不会在此处仍发生异常
            try:
                response = get_answer(messages)
                new_datum = {
                    'metaphor_id': entry["metaphor_id"],
                    'input': input
                }
                prediction = response.choices[0].message.content
                new_datum["prediction"] = prediction
                new_data.append(new_datum)
                if len(new_data) % 100 == 0:
                    json.dump(new_data,
                              open(f"../preprocess_data/temp_data/temp_data_{index + 1}.json", 'w', encoding="utf-8"),
                              indent=4, ensure_ascii=False)
                    index += 1
            except openai.RateLimitError as e:
                print("发送请求太频繁了！30s后尝试重连OpenAI...")
                time.sleep(30)
                response = get_answer(messages)
                new_datum = {
                    'metaphor_id': entry["metaphor_id"],
                    'input': input
                }
                prediction = response.choices[0].message.content
                new_datum["prediction"] = prediction
                new_data.append(new_datum)
            except Exception as e:
                json.dump(new_data, open(target_data_file, 'w', encoding="utf-8"), indent=4, ensure_ascii=False)
                print("遇到其他错误:", e)
        except Exception as e:
            # 若是遇到其他异常，立马保存已预测得到的数据
            json.dump(new_data, open(target_data_file, 'w', encoding="utf-8"), indent=4, ensure_ascii=False)
            print("遇到其他错误:", e)

    with open(target_data_file, "w") as f:
        json.dump(new_data, open(target_data_file, 'w', encoding="utf-8"), indent=4, ensure_ascii=False)