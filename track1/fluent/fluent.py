import jieba
import time
from models import MaskedBert
from tqdm import tqdm
import pandas as pd
import numpy as np

start_time = time.time()

model = MaskedBert.from_pretrained("./chinese_bert_wwm_ext_pytorch")

print(f"Loading ngrams model cost {time.time() - start_time:.3f} seconds.")


def read_data(path, src):
    data = []
    with open(path, "r", encoding="utf8") as fin:
        lines = fin.readlines()
    for line, src_line in zip(lines, src):
        src_list = list(src_line)
        sent = src_line
        items = line.strip().split(", ")
        if len(items) == 2:
            pass
        else:
            for i in range(1, len(items), 2):
                src_list[int(items[i]) - 1] = items[i + 1]
            sent = ''.join(src_list)
        data.append(sent)
    return data


def read_src(path):
    data = []
    with open(path, "r", encoding="utf8") as fin:
        lines = fin.readlines()
    for line in lines:
        items = line.strip().split("\t")
        data.append(items[1])
    return data


src_path = "yaclc-csc_test.src"
hyp_path = "yaclc-csc-test_78.5.lbl"
true_path = "true_0824.lbl"

src = read_src(src_path)
pred_data = read_data(hyp_path, src)
true_data = read_data(true_path, src)

count = 0
n = 0
df = pd.read_excel("track2_predict_boost3.xlsx")
src = df['origin']
pred_data = df['boost']
boost = []
threshold = 0.3
for s1, s2 in tqdm(zip(src, pred_data)):
    if s1 != 140:
        ppl1 = model.perplexity(
                                x=jieba.lcut(s1),  # 经过切词的句子或段落
                                verbose=False,  # 是否显示详细的probability，default=False
                                )
        ppl2 = model.perplexity(
                                x=jieba.lcut(s2),  # 经过切词的句子或段落
                                verbose=False,  # 是否显示详细的probability，default=False
                                )
    # if ppl1 - ppl2 > threshold and s2 != s1:
    #     print('保留')
    #     print(s1, ppl1)
    #     print(s2, ppl2)
    #     print(s3)
        if s2 != s1 and ppl2 - ppl1 > threshold:
            print('不纠')
            boost.append(s1)
            print(s1, ppl1)
            print(s2, ppl2)
        else:
            boost.append(s2)
    else:
        boost.append(s1)

df['boost'] = boost
df.to_excel("track2_predict_boost.xlsx")

