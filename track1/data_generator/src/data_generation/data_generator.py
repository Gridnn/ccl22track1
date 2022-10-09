import json
import os
import random
import re
import numpy
import requests
import numpy as np
from tqdm import tqdm
import threading
from jieba import posseg
import logging
import csv
import pickle
import pandas as pd
from openccpy import Opencc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Data_Generator(object):

    # 漏字：多字：形近字错误：音近字错误（前后鼻音，卷翘舍的错误）=1:1:1:3
    def __init__(self, save_path, _confusion_pronounce):
        """
        :param save_path:
        :param _confusion_pronounce: 自定义混淆集
        """
        self.probs_all = 0.8
        self.pinyin_prob = 0.5
        self.gongwen_pronounce_path = "../../data/confusion_pronounce_gongwen.json"
        self.gongwen_shape_path = "../../data/confusion_shape_gongwen.json"
        self.confusion_pronounce_path = "../../data/confusion_pronounce_tencent.json"
        self.confusion_shape_path = "../../data/confusion_shape_tencent.json"
        # self.extra_candidate_path = "../../data/extra_candidate.json"
        self.vocab_path = "../../data/vocab.txt"
        self.vocab_threshold = 100
        # self.extra_candidate = json.load(open(self.extra_candidate_path, 'r'))
        self.gongwen_pronoun_confusion = json.load(open(self.gongwen_pronounce_path, 'r'))
        self.gongwen_shape_confusion = json.load(open(self.gongwen_shape_path, 'r'))
        self.confusion_pronounce = json.load(open(self.confusion_pronounce_path, 'r'))
        self.confusion_shape = json.load(open(self.confusion_shape_path, 'r'))
        # self.confusion_pronounce = _confusion_pronounce

        self.stopword_path = "../../data/stop_words"
        self.not_common_path = "../../data/生僻字.txt"
        self.words_path = "../../data/words.txt"
        self.vocab, self.stopwords, self.words = self.load_data()
        self.mx1 = threading.Lock()
        headers = ['doc_idx', 'mode', 'error_sentence', 'start_offset', 'end_offset', 'error_word', 'correct_word',
                   'sentence_startoffset', 'sentence_endoffset', 'raw_sentence']
        self.csv_writer = csv.DictWriter(open(save_path, 'w', encoding='utf-8'), headers, delimiter='\t')
        self.csv_writer.writeheader()

        # self.tokenizer = jieba.lcut

    def load_data(self):
        stop_words, vocab = [], []
        with open(self.stopword_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                cont = line.strip('\n').strip()
                if self.is_chinese_char(cont):
                    if len(cont) < 2:
                        stop_words.append(cont)
                else:
                    stop_words.append(cont)
        non_common = [line.strip() for line in open(self.not_common_path, 'r', encoding='utf-8')]

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                cont = line.strip('\n').split('\t')
                if int(cont[1]) > self.vocab_threshold and cont[0] not in stop_words and cont[0] not in non_common:
                    if cont[0] != Opencc.to_simple(cont[0]):
                        vocab.append(Opencc.to_simple(cont[0]))
                    else:
                        vocab.append(cont[0])

        words = []
        with open(self.words_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                if len(line) <= 2:
                    words.append(line)

        return vocab, stop_words, words

    def data_generate(self, input_data):
        data_all = []
        for doc_idxes, (doc_idx, data) in enumerate(input_data):
            # logger.info('the process doc is {}'.format(doc_idx))
            if str(data) == 'nan':
                continue
            sentence_lst = self.cut_sentence(data)

            # 计算句子在文章中的总位移
            sentence_length = [len(sentence) for sentence in sentence_lst]
            start_offset_map = [0 for _ in range(len(sentence_lst))]
            for sentence_id, sentence in enumerate(sentence_lst):
                start_offset_map[sentence_id] = sum([sentence_length[i] for i in range(sentence_id)])

            # 将句子中的空格，制表符去除
            sentence_cleaned = self.sentence_cleaning(sentence_lst)
            if len(sentence_lst) != len(sentence_cleaned):
                continue

            # 如果句子较短，不进行mask， len(sentence) < 10
            masked_sentence_idx = []
            for idx, sentence in enumerate(sentence_cleaned):
                if len(sentence) > 20:
                    masked_sentence_idx.append(idx)

            masked_sentence_length = len(masked_sentence_idx)
            if masked_sentence_length == 0:
                continue
            # all
            mask_idx_len = int(np.round(len(masked_sentence_idx) * self.probs_all))
            pronounce_idx_len = int(np.ceil(mask_idx_len * self.pinyin_prob))
            res_count = mask_idx_len - pronounce_idx_len
            missing_idx_len, extra_idx_len, shape_idx_len = 0, 0, 0
            if res_count >= 3:
                missing_idx_len = int(np.round(res_count // 3))
                res_count = res_count - missing_idx_len
                extra_idx_len = res_count // 2
                shape_idx_len = res_count - extra_idx_len
            elif res_count > 0:
                while res_count > 0:
                    probs = np.random.randint(1, 3)
                    if probs == 1:
                        missing_idx_len += 1
                    elif probs == 2:
                        extra_idx_len += 1
                    else:
                        shape_idx_len += 1
                    res_count -= 1
            logger.info("raw sentence len {}, all data {}, pronounce data {}, missing data {}, extra data {}, shape data {}".format(masked_sentence_length,
                        mask_idx_len, pronounce_idx_len, missing_idx_len, extra_idx_len, shape_idx_len))

            # 生成音似句子
            res_mask_idx = masked_sentence_idx[:]
            non_used = []
            while pronounce_idx_len > 0 and res_mask_idx:
                idx = np.random.choice(res_mask_idx)
                sentence = sentence_cleaned[idx]
                sentence_offset = start_offset_map[idx]
                result = self.get_pronounce_sim_sentence(sentence)
                if result is not None and result['mode'] != "":
                    result["doc_idx"] = doc_idx
                    result["sentence_startoffset"] = sentence_offset
                    result["sentence_endoffset"] = sentence_offset + len(sentence_lst[idx])
                    result['raw_sentence'] = sentence_lst[idx]
                    data_all.append(result)
                    pronounce_idx_len -= 1
                    res_mask_idx.remove(idx)
                else:
                    non_used.append(idx)
                    res_mask_idx.remove(idx)
            if non_used:
                res_mask_idx.extend(non_used)
                res_mask_idx = list(set(res_mask_idx))

            while shape_idx_len > 0 and res_mask_idx:
                idx = np.random.choice(res_mask_idx)
                sentence = sentence_cleaned[idx]
                sentence_offset = start_offset_map[idx]
                result = self.get_shape_sim_sentence(sentence)
                if result is not None and result['mode'] != "":
                    result["doc_idx"] = doc_idx
                    result["sentence_startoffset"] = sentence_offset
                    result["sentence_endoffset"] = sentence_offset + len(sentence_lst[idx])
                    result['raw_sentence'] = sentence_lst[idx]
                    data_all.append(result)
                    shape_idx_len -= 1
                    res_mask_idx.remove(idx)
                else:
                    res_mask_idx.remove(idx)

            if len(data_all) == 1000:
                self.save_data(data_all)
                data_all = []
        if len(data_all) != 0:
            self.save_data(data_all)

    def get_pronounce_sim_sentence(self, sentence):
        item = {
            "mode": "",
            "error_sentence": '',
            "start_offset": 0,
            "end_offset": 0,
            "error_word": '',
            'correct_word': ''
        }
        _, word_mapping, mask_mapping = self.sentence_preprocess(sentence)
        mask_idx = list(np.where(mask_mapping == 1)[0])
        times = 6
        while times > -1:
            if not mask_idx:
                break
            if random.random() <= 0.15:
                # 选词
                res_mapping = [(i+1) * j for i, j in zip(word_mapping, mask_mapping)]
                word_span = []
                l_idx, r_idx = 0, 1
                while r_idx < len(res_mapping):
                    if res_mapping[l_idx] == res_mapping[r_idx]:
                        r_idx += 1
                    else:
                        if r_idx - l_idx == 1 or sum(res_mapping[l_idx:r_idx]) == 0:
                            l_idx = r_idx
                            r_idx += 1
                        else:
                            word_span.append((l_idx, r_idx))
                            l_idx = r_idx
                            r_idx += 1
                if not word_span:
                    continue
                word_span = random.choice(word_span)
                words = sentence[word_span[0]:word_span[1]]
                word_idxs = [i for i in range(len(words))]
                while len(word_idxs) > 0:
                    char_idx = np.random.choice(word_idxs)
                    candidate = list()
                    # candidate.extend(self.gongwen_pronoun_confusion.get(words[char_idx], ''))
                    candidate.extend(self.confusion_pronounce.get(words[char_idx], ''))
                    candidate = list(set(candidate))
                    char_offset = char_idx + word_span[0]
                    if candidate:
                        candi_word = random.choice(candidate)
                        item["mode"] = "pronounce_smi"
                        item["error_sentence"] = sentence[:char_offset] + candi_word + sentence[char_offset + 1:]
                        item["start_offset"] = char_offset
                        item["end_offset"] = char_offset + 1
                        item["error_word"] = candi_word
                        item["correct_word"] = sentence[char_offset]
                        return item
                    else:
                        word_idxs = list(word_idxs)
                        word_idxs.remove(char_idx)
                        times -= 1
            else:
                sen_idx = random.choice(mask_idx)
                word = sentence[sen_idx]
                candidate = []
                # 公文中的音似候选集
                # candidate.extend(self.gongwen_pronoun_confusion.get(word, ''))
                candidate.extend(self.confusion_pronounce.get(word, ''))
                candidate = list(set(candidate))
                if candidate:
                    candi_word = random.choice(candidate)
                    item["mode"] = "pronounce_smi"
                    item["error_sentence"] = sentence[:sen_idx] + candi_word + sentence[sen_idx + 1:]
                    item["start_offset"] = sen_idx
                    item["end_offset"] = sen_idx + 1
                    item["error_word"] = candi_word
                    item["correct_word"] = sentence[sen_idx]
                    return item
                else:
                    mask_idx = list(mask_idx)
                    mask_idx.remove(sen_idx)
                    times -= 1
        return None

    def sentence_cleaning(self, sentences):
        """
        删除句子中的制表符，空格等
        :param sentences:
        :return:
        """
        if isinstance(sentences, list):
            new_sentences = []
            for sentence in sentences:
                sentence = re.sub(r'\s', '', sentence)
                new_sentences.append(sentence)
            return new_sentences

        else:
            sentences = re.sub(r'\s', '', sentences)
            return sentences

    def get_shape_sim_sentence(self, sentence):
        item = {
            "mode": "",
            "error_sentence": '',
            "start_offset": 0,
            "end_offset": 0,
            "error_word": '',
            'correct_word': ''
        }
        words_lst, word_mapping, mask_mapping = self.sentence_preprocess(sentence)
        mask_idx = list(np.where(mask_mapping == 1)[0])

        times = 6
        while times >= 0:
            if len(mask_idx) < 0 or not mask_idx:
                break
            if random.random() <= 0.15:
                # 选词
                res_mapping = [(i+1) * j for i, j in zip(word_mapping, mask_mapping)]
                word_span = []
                l_idx, r_idx = 0, 1
                while r_idx < len(res_mapping):
                    if res_mapping[l_idx] == res_mapping[r_idx]:
                        r_idx += 1
                    else:
                        if r_idx - l_idx == 1 or sum(res_mapping[l_idx:r_idx]) == 0:
                            l_idx = r_idx
                            r_idx += 1
                        else:
                            word_span.append((l_idx, r_idx))
                            l_idx = r_idx
                            r_idx += 1
                if not word_span:
                    continue
                word_span = random.choice(word_span)
                words = sentence[word_span[0]:word_span[1]]
                word_idxs = [i for i in range(len(words))]
                while len(word_idxs) > 0:
                    char_idx = np.random.choice(word_idxs)
                    candidate = []
                    # candidate.extend(self.gongwen_shape_confusion.get(words[char_idx], ''))
                    candidate.extend(self.confusion_shape.get(words[char_idx], ''))
                    candidate = list(set(candidate))
                    char_offset = char_idx + word_span[0]
                    if candidate:
                        candi_word = random.choice(candidate)
                        item["mode"] = "shape_smi"
                        item["error_sentence"] = sentence[:char_offset] + candi_word + sentence[char_offset + 1:]
                        item["start_offset"] = char_offset
                        item["end_offset"] = char_offset + 1
                        item["error_word"] = candi_word
                        item["correct_word"] = sentence[char_offset]
                        return item
                    else:
                        word_idxs = list(word_idxs)
                        word_idxs.remove(char_idx)
                        times -= 1
            else:
                sen_idx = random.choice(mask_idx)
                _char = sentence[sen_idx]
                candidate = []
                # candidate.extend(self.gongwen_shape_confusion.get(_char, ''))
                candidate.extend(self.confusion_shape.get(_char, ''))
                candidate = list(set(candidate))
                if candidate:
                    cand_char = random.choice(candidate)
                    item["mode"] = "shape_smi"
                    item["error_sentence"] = sentence[:sen_idx] + cand_char + sentence[sen_idx + 1:]
                    item["start_offset"] = sen_idx
                    item["end_offset"] = sen_idx + 1
                    item["error_word"] = cand_char
                    item["correct_word"] = sentence[sen_idx]
                    return item
                else:
                    times -= 1
                    mask_idx = list(mask_idx)
                    mask_idx.remove(sen_idx)
        return None

    def sentence_preprocess(self, sentence, mask_last=False):
        """
        将句子中的停用词，实体，非中文字符过滤掉
        :param sentence:
        :param stop_words:
        :return: mask_flag_list [0, 0, 1, 0]; 0->不mask； 1-> 以概率mask
        """
        # 滤掉停用词， 分词， word_mapping
        word_lst, _, word_mapping = self.get_wordseg_result(sentence)
        masked_flag = np.ones(shape=(len(sentence,)), dtype=int)
        for idx, word in enumerate(word_lst):
            if word in self.stopwords:
                start_idx = word_mapping.index(idx)
                end_idx = len(word_mapping) - word_mapping[::-1].index(idx)
                masked_flag[start_idx: end_idx] = 0
            elif not self.is_chinese_char(word):
                start_idx = word_mapping.index(idx)
                end_idx = len(word_mapping) - word_mapping[::-1].index(idx)
                masked_flag[start_idx: end_idx] = 0

        # get ner
        thread_lst = []
        tmp_result = []

        if tmp_result:
            for result in tmp_result:
                masked_flag[result[1]:result[2]] = 0

        # 句尾不设错
        if mask_last:
            if masked_flag[-1] == 1:
                masked_flag[-1] = 0

        return word_lst, word_mapping, masked_flag

    def is_chinese_char(self, word):
        for w in word:
            if u'\u4e00' > w or w > u'\u9fff':
                return False
        return True

    def get_wordseg_result(self, text):
        """
        :param text:
        :return: {'wordseg_mapping': [0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11]}
        """
        wordseg_lst, wordpos_lst, inverse_idx_lst = [], [], []
        for idx, (word, pos) in enumerate(posseg.cut(text)):
            wordseg_lst.append(word)
            wordpos_lst.append(pos)
            inverse_idx_lst.extend([idx] * len(word))
        return wordseg_lst, wordpos_lst, inverse_idx_lst

    def cut_sentence(self, passage, splits=['。', '？', '\?', '！', '!']):
        matches = re.split("(" + '|'.join(splits) + ")", passage)
        values = matches[::2]
        delimiters = matches[1::2] + ['']
        pieces = [v + d for v, d in zip(values, delimiters)]
        result = []
        for piece in pieces:
            if len(piece) == 0: continue
            else:
                result.append(piece)
        return result

    def save_data(self, values):
        self.mx1.acquire()
        self.csv_writer.writerows(values)
        self.mx1.release()


class ThreadClass(threading.Thread):
    def __init__(self, target, args):
        threading.Thread.__init__(self)
        self.target = target
        self.args = args
        self.result = []

    def run(self):
        self.result = self.target(*self.args)

    def get_results(self):
        return self.result


if __name__ == "__main__":

    data_path = "/pretrain.csv"
    save_path = data_path.split('.')[0] + '_generated_data.csv'
    data = pd.read_csv(data_path)
    print(data.shape)
    print(data.columns)
    _count_max = 50
    recall_result = []

    generator = Data_Generator(save_path, {})

    threads_num = 80
    thread_lst = []
    sub_size = data.shape[0]//threads_num if data.shape[0] % threads_num == 0 else data.shape[0]//threads_num + 1

    for idx in range(threads_num):
        thread_lst.append(threading.Thread(target=generator.data_generate,
                                           args=(data[idx * sub_size:(idx+1)*sub_size if data.shape[0] > (idx+1) * sub_size else data.shape[0]],)))

    for thread in thread_lst:
        thread.setDaemon(True)
        thread.start()

    for thread in thread_lst:
        thread.join()


