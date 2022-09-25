# coding: utf-8
import os
import numpy as np
import re
import sys
import json
import codecs
from tqdm import tqdm
from collections import Counter
from multiprocessing import cpu_count, Pool
from itertools import zip_longest
import jieba_fast as jieba
import plyvel
from pypinyin import lazy_pinyin

"""input: *.txt的文件目录
   output: leveldb文件
"""
CHINESE_RE_PATTERN = r'\u3400-\u9FFF'

def format_pinyin(py):
    py = re.sub("sh|ch|zh", lambda x: x.group()[:1], py)
    py = re.sub("ang|eng|ing|ong", lambda x: x.group()[:2], py)
    return py


def preprocess_and_split_sentences(input_text: str):
    norm_mapping = [(i, i) for i in range(len(input_text))]
    sentence_lst = []
    for search_item in re.finditer(r'[%sA-Z“”\"]+' % CHINESE_RE_PATTERN, input_text):
        start, end = search_item.start(), search_item.end()
        sentence_lst.append((input_text[start:end], norm_mapping[start:end]))

    return sentence_lst


input_dir = '../txt'
output_dir = '../leveldb'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
files = [file for file in files if file.endswith("txt") and os.path.isfile(file)]
file_chunks = np.array_split(files, len(files) // 200)


def gen_dicts(file_chunk, curr_id, total_count):
    py_dict = dict()
    word_freq = Counter()
    union_word_freq = Counter()
    for file in tqdm(file_chunk, desc='Gen dicts {}/{}'.format(curr_id, total_count)):
        fp = codecs.open(file, encoding="utf-8")
        lines = fp.readlines()
        fp.close()
        for line in lines:
            line = line.replace('\\n', '').strip()
            if not line: continue
            sentence_lst = preprocess_and_split_sentences(line)
            sentences = [item[0] for item in sentence_lst]
            if len(sentences) < 5: continue
            for sentence in sentences:
                former = 'S'
                word_freq['S'] = word_freq.get('S', 0) + 1
                cut_list = jieba.cut(sentence)
                for word in cut_list:
                    word = word.strip()
                    if not word: continue
                    pinyin = format_pinyin(''.join(lazy_pinyin(word)))
                    if pinyin not in py_dict:
                        py_dict[pinyin] = Counter()
                    py_dict[pinyin][word] = py_dict[pinyin].get(word, 0) + 1
                    union_word = former + '_' + word
                    word_freq[word] = word_freq.get(word, 0) + 1
                    union_word_freq[union_word] = union_word_freq.get(union_word, 0) + 1
                    former = word
                if former == 'S': continue
                union_word = former + '_E'
                word_freq['E'] = word_freq.get('E', 0) + 1
                union_word_freq[union_word] = union_word_freq.get(union_word, 0) + 1
    return py_dict, Counter(word_freq), Counter(union_word_freq)

# Generate Frequency Count in Parallel
pool = Pool(processes=cpu_count())
pool_result = []
for i in range(len(file_chunks)):
    pool_result.append(pool.apply_async(gen_dicts, (file_chunks[i], i, len(file_chunks))))
pool.close()
pool.join()

# Reduce Results
py_dict, word_freq, union_word_freq = {}, Counter(), Counter()
for result in tqdm(pool_result, desc="Reduce"):
    sub_py, sub_word_freq, sub_union_word_freq = result.get()
    for k, v in sub_py.items():
        if k not in py_dict:
            py_dict[k] = Counter()
        py_dict[k] += v
    word_freq += sub_word_freq
    union_word_freq += sub_union_word_freq

# Save LevelDB
py_db = plyvel.DB(os.path.join(output_dir, "py_db"), create_if_missing=True)
for idx, py in enumerate(tqdm(py_dict, desc="Save Pinyin LevelDB")):
    sub_py_db = py_db.prefixed_db(bytes((py + "::").encode("utf-8")))
    with sub_py_db.write_batch(transaction=True) as wb:
        for w, c in py_dict[py].items():
            wb.put(bytes(w.encode("utf-8")), bytes(str(c).encode("utf-8")))

word_freq_db = plyvel.DB(os.path.join(output_dir, "word_freq_db"), create_if_missing=True)
with word_freq_db.write_batch(transaction=True) as wb:
    for idx, (w, c) in enumerate(tqdm(word_freq.items(), desc="Save Word Freq LevelDB")):
        wb.put(bytes(w.encode("utf-8")), bytes(str(c).encode("utf-8")))

with word_freq_db.write_batch(transaction=True) as wb:
    for idx, (w, c) in enumerate(tqdm(union_word_freq.items(), desc="Save Union Freq LevelDB")):
        wb.put(bytes("".join(w.split("_")).encode("utf-8")), bytes(str(c).encode("utf-8")))