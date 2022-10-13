#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
fork https://github.com/hiyoung123/YoungCorrector/blob/master/corrector.py
删简版-删掉自定义词对混淆词典/未登录词检测，只保留字词纠错
"""

import codecs
import jieba
import kenlm
import numpy as np
import pandas as pd
import operator
import os
import time
from pypinyin import lazy_pinyin

from loguru import logger
from src.text_utils import *


class ErrorType(object):
    word = 'word'
    char = 'char'


class config:
    same_pinyin_path = "../data/same_pinyin.txt"
    same_stroke_path = "../data/same_stroke.txt"
    # lm_model_path = "../data/people_chars_lm.klm"
    lm_model_path = '../model/ngram_model/text.klm'
    char_set_path = "../data/common_char_set.txt"
    pinyin2word_path = "../data/pinyin2word.model"


class NgramCorrector(object):

    def __init__(self, config):
        self.same_pinyin_path = config.same_pinyin_path
        self.same_stroke_path = config.same_stroke_path
        self.char_set_path = config.char_set_path
        self.lm_model_path = config.lm_model_path
        self.pinyin2word_path = config.pinyin2word_path

        self.same_pinyin_dict = self._load_same_pinyin_dict(self.same_pinyin_path)
        self.same_stroke_dict = self._load_same_stroke_dict(self.same_stroke_path)
        self.pinyin2word = self._load_pinyin_2_word(self.pinyin2word_path)
        self.char_set = self._load_char_set(self.char_set_path)
        self.tokenizer = jieba
        self.lm = kenlm.Model(self.lm_model_path)

    def _load_same_pinyin_dict(self, path, sep='\t'):
        result = dict()
        if not os.path.exists(path):
            logger.warn("file not exists: %s" % path)
            return result
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 2:
                    key_char = parts[0]
                    same_pron_same_tone = set(list(parts[1]))
                    same_pron_diff_tone = set(list(parts[2]))
                    value = same_pron_same_tone.union(same_pron_diff_tone)
                    if key_char and value:
                        result[key_char] = value
        self.same_pinyin_path = path
        return result

    def _load_same_stroke_dict(self, path,  sep='\t'):
        result = dict()
        if not os.path.exists(path):
            logger.warn("file not exists: %s" % path)
            return result
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 1:
                    for i, c in enumerate(parts):
                        result[c] = set(list(parts[:i] + parts[i + 1:]))
        self.same_stroke_path = path
        return result

    def _load_pinyin_2_word(self, path):
        result = dict()
        if not os.path.exists(path):
            logger.warn("file not exists: %s" % path)
            return result
        with codecs.open(path, 'r', encoding='utf-8') as f:
            a = f.read()
            result = eval(a)
        return result

    def _load_char_set(self, path):
        words = set()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w.startswith('#'):
                    continue
                if w:
                    words.add(w)
        return words

    def _check_state(self):
        res = True
        res &= self.same_pinyin_dict is not None
        res &= self.same_stroke_dict is not None
        return res

    def _process_text(self, text):
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        text = uniform(text)
        return text

    def _check_in_errors(self, maybe_errors, maybe_err):
        error_word_idx = 0
        begin_idx = 1
        end_idx = 2
        for err in maybe_errors:
            if maybe_err[error_word_idx] in err[error_word_idx] and maybe_err[begin_idx] >= err[begin_idx] and \
                    maybe_err[end_idx] <= err[end_idx]:
                return True
        return False

    def _get_max_len(self, d):
        return max(map(len, [w for w in d]))

    def FMM(self, word_dict, token, window_size=5):
        idxs = []
        result = []
        index = 0
        text_size = len(token)
        while text_size > index:
            for size in range(window_size + index, index, -1):
                piece = token[index:size]
                if piece in word_dict:
                    index = size - 1
                    idxs.append(index-len(piece)+1)
                    result.append(piece)
                    break
            index = index + 1
        return idxs, result

    def _is_filter_token(self, token):
        # 空
        if not token.strip():
            return True
        # 全是英文
        if is_alphabet_string(token):
            return True
        # 全是数字
        if token.isdigit():
            return True
        # 只有字母和数字
        if is_alp_diag_string(token):
            return True
        # 过滤标点符号
        if re_poun.match(token):
            return True

        return False

    def _get_maybe_error_index(self, scores, ratio=0.6745, threshold=2.0):
        """
        取疑似错字的位置，通过平均绝对离差（MAD）
        :param scores: np.array
        :param ratio: 正态分布表参数
        :param threshold: 阈值越小，得到疑似错别字越多
        :return: 全部疑似错误字的index: list
        """
        result = []
        scores = np.array(scores)
        if len(scores.shape) == 1:
            scores = scores[:, None]
        median = np.median(scores, axis=0)  # get median of all scores
        margin_median = np.abs(scores - median).flatten()  # deviation from the median
        # 平均绝对离差值
        med_abs_deviation = np.median(margin_median)
        if med_abs_deviation == 0:
            return result
        y_score = ratio * margin_median / med_abs_deviation
        # 打平
        scores = scores.flatten()
        maybe_error_indices = np.where((y_score > threshold) & (scores < median))
        # 取全部疑似错误字的index
        result = list(maybe_error_indices[0])
        return result

    def _detect_by_word_ngrm(self, maybe_errors, sentence, start_idx):
        try:
            ngram_avg_scores = []
            tokens = [x for x in self.tokenizer.cut(sentence)]
            print(tokens)
            for n in [1, 2, 3]:
                scores = []
                for i in range(len(tokens) - n + 1):
                    word = tokens[i:i + n]
                    score = self.lm.score(' '.join(list(word)), bos=False, eos=False)
                    scores.append(score)
                if not scores:
                    continue
                # 移动窗口补全得分
                for _ in range(n - 1):
                    scores.insert(0, scores[0])
                    scores.append(scores[-1])
                    # scores.append(sum(scores)/len(scores))
                avg_scores = [sum(scores[i:i + n]) / len(scores[i:i + n]) for i in range(len(tokens))]
                ngram_avg_scores.append(avg_scores)

            if ngram_avg_scores:
                # 取拼接后的n-gram平均得分
                sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
                # 取疑似错字信息
                for i in self._get_maybe_error_index(sent_scores, threshold=2.0):
                    token = tokens[i]
                    # i = sentence.find(token)
                    if len(token) == 1:
                        type = ErrorType.char
                    else:
                        type = ErrorType.word
                    maybe_err = [token, i+start_idx, i+len(token)+start_idx, type]
                    if maybe_err not in maybe_errors and not self._check_in_errors(maybe_errors, maybe_err):
                        maybe_errors.append(maybe_err)

        except IndexError as ie:
            logger.warn("index error, sentence:" + sentence + str(ie))
        except Exception as e:
            logger.warn("detect error, sentence:" + sentence + str(e))

    def _detect_by_char_ngrm(self, maybe_errors, sentence, start_idx):
        try:
            ngram_avg_scores = []
            for n in [1, 2, 3, 4]:
                scores = []
                for i in range(len(sentence) - n + 1):
                    word = sentence[i:i + n]
                    score = self.lm.score(' '.join(list(word)), bos=False, eos=False)
                    scores.append(score)
                if not scores:
                    continue
                # 移动窗口补全得分
                for _ in range(n - 1):
                    scores.insert(0, scores[0])
                    scores.append(scores[-1])
                    # scores.append(sum(scores) / len(scores))
                avg_scores = [sum(scores[i:i + n]) / len(scores[i:i + n]) for i in range(len(sentence))]
                ngram_avg_scores.append(avg_scores)

            if ngram_avg_scores:
                # 取拼接后的n-gram平均得分
                sent_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
                # 取疑似错字信息
                for i in self._get_maybe_error_index(sent_scores, threshold=1.7):
                    token = sentence[i]
                    maybe_err = [token, i+start_idx, i+len(token)+start_idx, ErrorType.char]
                    if maybe_err not in maybe_errors and not self._check_in_errors(maybe_errors, maybe_err):
                        maybe_errors.append(maybe_err)

        except IndexError as ie:
            logger.warn("index error, sentence:" + sentence + str(ie))
        except Exception as e:
            logger.warn("detect error, sentence:" + sentence + str(e))

    def _detect_short(self, sentence, start_idx):
        maybe_errors = []
        if not sentence.strip():
            return maybe_errors
        self._detect_by_word_ngrm(maybe_errors, sentence, start_idx)
        self._detect_by_char_ngrm(maybe_errors, sentence, start_idx)
        return sorted(maybe_errors, key=lambda x: x[1], reverse=False)

    def _candidates(self, word, fregment=1):
        candidates = []
        if len(word) > 1:
            candidates += self._candidates_by_edit(word)
        candidates += self._candidates_by_pinyin(word)
        candidates += self._candidates_by_stroke(word)
        return list(set(candidates))

    def known(self, words):
        """The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.word_dict)

    def edits1(self, word):
        """All edits that are one edit away from `word`."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.char_set]
        return set(transposes + replaces)

    def _candidates_by_edit(self, word):
        return [w for w in self.known(self.edits1(word)) or [word] if lazy_pinyin(word) == lazy_pinyin(w)]

    def _candidates_by_pinyin(self, word):
        l = []
        r = list(self.pinyin2word.get(','.join(lazy_pinyin(word)), {word:''}).keys())
        for i, w in enumerate(word):
            before = word[:i]
            after = word[i+1:]
            a = list(self.same_pinyin_dict.get(w, w))
            l += [before+x+after for x in a]

        return set(l + r)

    def _candidates_by_stroke(self, word):
        l = []
        for i, w in enumerate(word):
            before = word[:i]
            after = word[i + 1:]
            a = list(self.same_stroke_dict.get(w, w))
            l += [before + x + after for x in a]

        return set(l)

    def _calibration(self, maybe_errors):
        res = []
        pre_item = None
        for cur_item, begin_idx, end_idx, err_type in maybe_errors:

            if pre_item is None:
                pre_item = [cur_item, begin_idx, end_idx, err_type]
                res.append(pre_item)
                continue
            if ErrorType.char == err_type and err_type == pre_item[3] and begin_idx == pre_item[2]:
                pre_item = [pre_item[0]+cur_item, pre_item[1], end_idx, ErrorType.word]
                res.pop()
            else:
                pre_item = [cur_item, begin_idx, end_idx, err_type]
            res.append(pre_item)
        return res

    def get_lm_correct_item(self, cur_item, candidates, before_sent, after_sent, threshold=58):
        """
        通过语言模型纠正字词错误
        :param cur_item: 当前词
        :param candidates: 候选词
        :param before_sent: 前半部分句子
        :param after_sent: 后半部分句子
        :param threshold: ppl阈值, 原始字词替换后大于ppl则是错误
        :return: str, correct item, 正确的字词
        """
        result = cur_item
        if cur_item not in candidates:
            candidates.append(cur_item)
        ppl_scores = {i: self.lm.perplexity(' '.join(list(before_sent + i + after_sent))) for i in candidates}
        sorted_ppl_scores = sorted(ppl_scores.items(), key=lambda d: d[1])
        # 增加正确字词的修正范围，减少误纠
        top_items = []
        top_score = 0.0
        for i, v in enumerate(sorted_ppl_scores):
            v_word = v[0]
            v_score = v[1]
            if i == 0:
                top_score = v_score
                top_items.append(v_word)
            # 通过阈值修正范围
            elif v_score < top_score + threshold:
                top_items.append(v_word)
            else:
                break
        if cur_item not in top_items:
            result = top_items[0]
        return result

    def correct(self, text):
        if text is None or not text.strip():
            logger.warn("Input text is error.")
            return text

        if not self._check_state():
            logger.warn("Corrector not init.")
            return text

        text_new = ''
        details = []
        text = self._process_text(text)
        blocks = split_long_text(text, include_symbol=True)
        for blk, idx in blocks:
            maybe_errors = self._detect_short(blk, idx)
            maybe_errors = self._calibration(maybe_errors)
            for cur_item, begin_idx, end_idx, err_type in maybe_errors:
                # 纠错，逐个处理
                before_sent = blk[:(begin_idx - idx)]
                after_sent = blk[(end_idx - idx):]

                # 取得所有可能正确的词
                candidates = self._candidates(cur_item)
                if not candidates:
                    continue
                corrected_item = self.get_lm_correct_item(cur_item, candidates, before_sent, after_sent)
                if corrected_item != cur_item:
                    blk = before_sent + corrected_item + after_sent
                    detail_word = [cur_item, corrected_item, begin_idx, end_idx]
                    details.append(detail_word)
            text_new += blk
        details = sorted(details, key=operator.itemgetter(2))
        return [{"error_correction": [cur[1]], "origin": cur[0], "start_offset": cur[2], "end_offset": cur[3]-1} for cur in details]


if __name__ == "__main__":

    corrector = NgramCorrector(config)
    text = ["满不满意？我听说中国化了很多很多钱，所以我想知道到底值不值得。"]
    start = time.time()
    result = []
    for _index, x in enumerate(text):
        y = corrector.correct(x)
        result.append(y)
        print(y)
    print(time.time() - start)
