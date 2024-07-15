# --coding:utf-8--
import pandas as pd
import json

# csv文件转化为json文件
def csv_to_json_test(csv_file, json_file):
    df = pd.read_csv(csv_file, encoding='utf-8')
    # 检查每一列中的空值数量
    # print(df.isnull().sum())
    data = []
    for index, row in df.iterrows():
        tenor_a, tenor_b, tenor_c, tenor_d = row['答案A_本体'].strip(), row['答案B_本体'].strip(), row['答案C_本体'].strip(), row['答案D_本体'].strip()
        vehicle_a, vehicle_b, vehicle_c, vehicle_d = row['答案A_喻体'].strip(), row['答案B_喻体'].strip(), row['答案C_喻体'].strip(), row['答案D_喻体'].strip()
        ground_a, ground_b, ground_c, ground_d = row['答案A_共性'].strip(), row['答案B_共性'].strip(), row['答案C_共性'].strip(), row['答案D_共性'].strip()
        option_A = ("[" + tenor_a + ", " + vehicle_a + ", " + ground_a + "]").replace("\n", "\t")
        option_B = ("[" + tenor_b + ", " + vehicle_b + ", " + ground_b + "]").replace("\n", "\t")
        option_C = ("[" + tenor_c + ", " + vehicle_c + ", " + ground_c + "]").replace("\n", "\t")
        option_D = ("[" + tenor_d + ", " + vehicle_d + ", " + ground_d + "]").replace("\n", "\t")
        input_text = row['比喻句'].strip() + "\n" + "A: " + option_A + "\n" + "B: " + option_B + "\n" + "C: " + option_C + "\n" + "D: " + option_D + "\n"
        data.append({
            "metaphor_id": index,
            "input": input_text,
            "output": ""
        })
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def csv_to_json_valid(csv_file, json_file):
    df = pd.read_csv(csv_file, encoding='utf-8')
    # 检查每一列中的空值数量
    # print(df.isnull().sum())
    data = []
    for index, row in df.iterrows():
        tenor_a, tenor_b, tenor_c, tenor_d = row['答案A_本体'].strip(), row['答案B_本体'].strip(), row['答案C_本体'].strip(), row['答案D_本体'].strip()
        vehicle_a, vehicle_b, vehicle_c, vehicle_d = row['答案A_喻体'].strip(), row['答案B_喻体'].strip(), row['答案C_喻体'].strip(), row['答案D_喻体'].strip()
        ground_a, ground_b, ground_c, ground_d = row['答案A_共性'].strip(), row['答案B_共性'].strip(), row['答案C_共性'].strip(), row['答案D_共性'].strip()
        option_A = ("[" + tenor_a + ", " + vehicle_a + ", " + ground_a + "]").replace("\n", "\t")
        option_B = ("[" + tenor_b + ", " + vehicle_b + ", " + ground_b + "]").replace("\n", "\t")
        option_C = ("[" + tenor_c + ", " + vehicle_c + ", " + ground_c + "]").replace("\n", "\t")
        option_D = ("[" + tenor_d + ", " + vehicle_d + ", " + ground_d + "]").replace("\n", "\t")
        right_answer = row['正确答案']
        if right_answer == 'A':
            right_answer = right_answer + ": " + option_A
        elif right_answer == 'B':
            right_answer = right_answer + ": " + option_B
        elif right_answer == 'C':
            right_answer = right_answer + ": " + option_C
        elif right_answer == 'D':
            right_answer = right_answer + ": " + option_D
        else:
            raise ValueError("The right answer is not A, B, C or D.")
        input_text = row['比喻句'].strip() + "\n" + "A: " + option_A + "\n" + "B: " + option_B + "\n" + "C: " + option_C + "\n" + "D: " + option_D + "\n"
        data.append({
            "metaphor_id": index,
            "input": input_text,
            "output": right_answer
        })
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 转化为json文件
csv_to_json_test('../data/track2_test.csv', '../data/test.json')
csv_to_json_valid('../data/track2_validation.csv', '../data/valid.json')
