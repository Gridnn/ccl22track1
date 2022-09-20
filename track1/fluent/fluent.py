import jieba
import time
from models import MaskedBert
from tqdm import tqdm

start_time = time.time()

model = MaskedBert.from_pretrained("bert-base-chinese")

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


src_path = "../data/yaclc-csc_test.src"
hyp_path = "../data/decode/yaclc-csc-test.lbl"

src = read_src(src_path)
pred_data = read_data(hyp_path, src)

count = 0
n = 0
boost = []
threshold = 0.3

f1 = open(hyp_path)
lines = f1.readlines()

f2 = open("../data/decode/yaclc-csc-test_fluent.lbl", "w")
for s1, s2, line in tqdm(zip(src, pred_data, lines)):
    ppl1 = model.perplexity(x=jieba.lcut(s1),  # 经过切词的句子或段落
                            verbose=False,  # 是否显示详细的probability，default=False
                            )
    ppl2 = model.perplexity(x=jieba.lcut(s2),  # 经过切词的句子或段落
                            verbose=False,  # 是否显示详细的probability，default=False
                            )
    if s2 != s1 and ppl2 - ppl1 > threshold:
        print('不纠')
        print(s1, ppl1)
        print(s2, ppl2)

        id = line.split()[0]
        f2.write(id + '\t' + '0' + '\n')
    else:
        f2.write(line)

