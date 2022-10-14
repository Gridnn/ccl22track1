# coding: utf-8

import sys
import os
import json
import re
from itertools import product
from tqdm import tqdm

from ner_model import LMModel

ENTITY_DICT_PATH = "./entity_dict.json"
CONFUSION_DICT_PATH = "./whole_confusion.json"
NER_MODEL_PATH = "./model_cluener_crf"
MAX_OPERATION = 2.5e8


def read_json(json_file):
    with open(json_file, encoding='utf-8') as f:
        data_dict: dict = json.load(f)
    return {k: set(v) for k, v in data_dict.items()}


def reject_corrections(text_trg: list, text_entity_info: list, overlap=False):
    # text_trg = [(pos, correction, )]; text_entity_info = [(entity_type, start, end, length)]
    if (not text_trg) or (not text_entity_info):
        return text_trg

    def valid_trg_nonoverlap(trg, text_entity_info):
        trg_pos = int(trg[0])
        for entity_type, start, end, length in text_entity_info:
            if start <= trg_pos <= end:
                return False
        return True

    def valid_trg_overlap(trg, text_entity_info):
        trg_pos = int(trg[0])
        trg_type = trg[2]["tag"]
        trg_start = trg[2]["start"] + 1
        trg_end = trg[2]["end"]
        trg_len = trg[2]["end"] - trg[2]["start"]
        for entity_type, start, end, length in text_entity_info:
            if start <= trg_pos <= end:
                if (start < trg_start < end < trg_end) or (trg_start < start < trg_end < end):
                    return False
                if trg_len <= length:
                    return False
        return True

    valid_trg = [valid_trg_nonoverlap, valid_trg_overlap]

    text_trg = list(filter(lambda x: valid_trg[int(overlap)](x, text_entity_info), text_trg))
    return text_trg


def recall_corrections(pred: dict, confusion_dict: dict, entity_dict: dict, entity_char_set: set):
    print(pred, flush=True)
    entity_set = entity_dict[pred["tag"]]
    if not entity_set:
        return []
    for char in pred["value"]:
        char_candidates = [char] + confusion_dict.get(char, [])
        for char_candidate in char_candidates:
            if char_candidate in entity_char_set:
                break
        else:
            return []
    char_candidate_sizes = [len(confusion_dict.get(char, [])) + 1 for char in pred["value"]]  # +1是char自身
    candidate_size = 1
    for char_candidate_size in char_candidate_sizes:
        candidate_size *= char_candidate_size
    if candidate_size > MAX_OPERATION:
        return []
    pred_candidates = [list(range(char_candidate_size)) for char_candidate_size in char_candidate_sizes]
    pred_value = list(pred["value"])
    top_candidate = None
    min_edit_count = float("inf")
    for candidate in product(*pred_candidates):
        edit_count = 0
        temp_pred_value = pred_value.copy()
        for i, char_cand in enumerate(candidate):
            if char_cand == 0:  # 表示不改动
                continue
            edit_count += 1
            temp_pred_value[i] = confusion_dict[pred_value[i]][char_cand - 1]
        temp_pred_value = ''.join(temp_pred_value)
        if temp_pred_value not in entity_set:
            continue
        if edit_count >= min_edit_count:
            continue
        top_candidate = temp_pred_value
        min_edit_count = edit_count

    if top_candidate is None:
        return []
    start = pred["start"] + 1
    i = 0
    corrections = []
    for o, c in zip(pred["value"], top_candidate):
        if o != c:
            corrections.append((str(start + i), c, pred))
        i += 1
    return corrections


def main(src_file: str, trg_file: str, output_file: str):
    entity_dict = read_json(ENTITY_DICT_PATH)  # entitytype: set(entities)
    entities = []
    for v in entity_dict.values():
        entities += list(v)
    entity_char_set = set(''.join(entities))
    entity_re_dict = {k: re.compile(
        '|'.join(v).replace('.', '\.').replace('+', '\+').replace('?', '\?').replace('*', '\*').replace('-',
                                                                                                        '\-').replace(
            '[', '\[')
        ) for k, v in entity_dict.items()}  # entitytype: repattern(entities)
    confusion_dict = read_json(CONFUSION_DICT_PATH)  # confusion set
    for k, v in confusion_dict.items():
        v.discard(k)
        confusion_dict[k] = sorted(v)
    # src file
    src_data = []
    with open(src_file, encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip()
            src_data.append(tuple(line.split('\t')))
            line = f.readline()
    src_ids = [src[0] for src in src_data]
    # trg file
    trg_data = []
    with open(trg_file, encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip()
            _id, trg = line.split(', ', maxsplit=1)
            if trg == '0':
                trg_data.append((_id, []))
            else:
                trg_lst = trg.split(', ')
                trg_lst = list(zip(trg_lst[::2], trg_lst[1::2]))
                trg_data.append((_id, trg_lst))
            line = f.readline()
    trg_ids = [trg[0] for trg in trg_data]
    assert src_ids == trg_ids
    # 实体词典匹配
    entity_dict_ret = []
    for _id, text in tqdm(src_data, desc='实体词典匹配'):
        text_ret = []
        for entity_type, entity_pattern in entity_re_dict.items():
            for match in re.finditer(entity_pattern, text):
                text_ret.append((entity_type, match.start() + 1, match.end(), match.end() - match.start()))
        entity_dict_ret.append(text_ret)
    # 命中部分保留原文，不采纳纠错结果
    for i, (_id, trg_lst) in enumerate(tqdm(trg_data, desc='命中部分保留原文，不采纳纠错结果')):
        trg_data[i] = (_id, reject_corrections(trg_lst, entity_dict_ret[i]))

    # ner模型预测
    with open(os.path.join(NER_MODEL_PATH, "config.json"), encoding='utf-8') as f:
        params = json.load(f)
    print(params)
    model = LMModel(config=params, logger=None)
    print(model.get_config())
    model.load_model()
    text_lst = [src[1] for src in src_data]
    entity_preds = model.predict_batch(text_lst)
    # 过滤已在词典中的实体
    for i, text_pred in enumerate(entity_preds):
        entity_preds[i] = list(filter(lambda x: x["value"] not in entity_dict[x["tag"]], text_pred))
    # 混淆集匹配词典召回实体
    recall_trg_ret = []  # [[(pos, correction, entity_pred), ...], ...]
    for text_pred in tqdm(entity_preds, desc='混淆集匹配词典召回实体'):
        # for text_pred in entity_preds:
        text_recall_trg = []
        for pred in text_pred:
            text_recall_trg += recall_corrections(pred, confusion_dict, entity_dict, entity_char_set)
        recall_trg_ret.append(text_recall_trg)
    # 实体重叠校验
    for i, recall_trg_lst in enumerate(tqdm(recall_trg_ret, desc='实体重叠校验')):
        recall_trg_ret[i] = reject_corrections(recall_trg_lst, entity_dict_ret[i], overlap=True)
    # 合并纠错结果
    recall_trg_ret = [[(j[0], j[1]) for j in i] for i in recall_trg_ret]
    final_trg_ret = []
    for (_id, trg_lst), recall_trg_lst in zip(trg_data, recall_trg_ret):
        text_trg_lst = trg_lst + recall_trg_lst
        if text_trg_lst:
            text_trg_lst = sorted(set(text_trg_lst), key=lambda x: int(x[0]))
        final_trg_ret.append((_id, text_trg_lst))
    # 输出到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (_id, text_trg_lst) in enumerate(tqdm(final_trg_ret, desc='输出到文件')):
            message = [_id]
            if not text_trg_lst:
                message.append('0')
            else:
                for pos, correction in text_trg_lst:
                    message += [pos, correction]
            message = ', '.join(message)
            f.write(message)
            if i != len(final_trg_ret) - 1:
                f.write('\n')


if __name__ == '__main__':
    src_file, trg_file, output_file = sys.argv[1:]
    main(src_file, trg_file, output_file)

# python3 ner.py ..data/yaclc-csc_test.src ..data/decode/yaclc-csc-test.lbl ..data/decode/yaclc-csc-test_ner.lbl
