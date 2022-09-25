# coding: utf-8
import os
from itertools import product
from functools import lru_cache
import re
import sys
import numpy as np
import plyvel
import math
from pypinyin import lazy_pinyin, pinyin, Style
from jieba_fast import Tokenizer, posseg
from Levenshtein import distance

semantic_data_dir = "./static"
leveldb_dir = "./model/leveldb"

CAND_ORIGINAL_WEIGHT = 1.0
CAND_HOMONYMS_WEIGHT = 0.5  # 1.1
CAND_STROKES_WEIGHT = 0.4  # 1.1
CAND_FREQ1_THRED = 5
CHINESE_RE_PATTERN = r'\u3400-\u9FFF'


class Candidate:
    def __init__(self, candidate_str, is_original=False, prev_candidate=None, proba=0.0, is_flag=False):
        self.word = candidate_str
        self.is_original = is_original
        self.prev_candidate = prev_candidate
        self.proba = proba
        self.is_flag = is_flag

    def __str__(self):
        return "<%s: %.2f>" % ("".join(self.word), self.proba)


class CandidateList:
    def __init__(self):
        self._candidate_list = [[Candidate("S", is_flag=True)]]
        self.origin_sequence = None

    def __getitem__(self, key):
        if len(self._candidate_list) <= key + 1:
            self._candidate_list.extend([[]] * (key + 2 - len(self._candidate_list)))
        return self._candidate_list[key+1]


class CandidateGen(object):
    def __init__(self, py_db, freq_db):
        self._py_db = py_db
        self.py_db_dict = {}
        self.prefix_db = {}
        with self._py_db.iterator() as it:
            for k, v in it:
                v = int(v)
                if v < CAND_FREQ1_THRED: continue
                self.py_db_dict[k] = v
                py, word = k.decode("utf-8").split("::")
                if py not in self.prefix_db: self.prefix_db[py] = {}
                self.prefix_db[py][word] = v
        self.shape_like_word2id, self.shape_like_id2group = self.load_shape_like()
        self.freq_db = freq_db

    @staticmethod
    def load_shape_like():
        user_dict_file_path = os.path.join(semantic_data_dir, "word_simi", "default_simi_word.txt")
        user_dict_file = open(user_dict_file_path, encoding="utf-8")
        user_dict_lst = [line.strip() for line in user_dict_file if line.strip()]
        user_dict_file.close()
        word2id = {}
        id2group = {}
        for idx, line in enumerate(user_dict_lst):
            line = line.strip()
            if line:
                words = {w for w in line.split(",") if w}
                id2group[idx] = words
                for w in words:
                    word2id[w] = idx
        return word2id, id2group

    @lru_cache(maxsize=int(os.environ.get("NGram_CandidateGen_CacheSize", "2000")))
    def add_candidates(self, word):
        raw_py = self.get_pinyin(word)
        py = format_pinyin(raw_py)  # normalize
        py = bytes(py.encode("utf-8"))
        sub_homonyms = {}
        for idx, char in enumerate(word):
            if char == '幺':
                print(char)
            if char in self.shape_like_word2id:
                cand_chars = self.shape_like_id2group[self.shape_like_word2id[char]] - {char}
                for sub_char in cand_chars:
                    sub_word = ''.join(word[:idx] + sub_char + word[idx+1:])
                    if int(self.freq_db.get(sub_word.encode('utf-8'), 0)) >= CAND_FREQ1_THRED:
                        sub_homonyms[sub_word] = CAND_STROKES_WEIGHT
        possible_py = product(*pinyin(word, style=Style.NORMAL, heteronym=False))
        for ppy in possible_py:
            ppy = format_pinyin("".join(ppy))
            if ppy not in self.prefix_db: continue
            for k, v in self.prefix_db[ppy].items():
                if self.get_pinyin(k) == raw_py:
                    sub_homonyms[k] = CAND_HOMONYMS_WEIGHT
                else:
                    sub_homonyms[k] = CAND_STROKES_WEIGHT
        return sub_homonyms

    @lru_cache(maxsize=int(os.environ.get("NGram_CandidateGen_CacheSize", "2000")))
    def get_pinyin(self, word):
        result = "".join(lazy_pinyin(word))
        return result

    def gen_candidates(self, word, raw_only):
        if len(word) == 0: return {}
        if len(word) == 1:  return {word: CAND_ORIGINAL_WEIGHT}
        # original
        candidates = {word: CAND_ORIGINAL_WEIGHT}
        if raw_only: return candidates
        # homonyms
        for homonym, weight in self.add_candidates(word).items():
            if homonym != word:
                candidates[homonym] = weight
        return candidates


class NgramCorrector(object):
    def __init__(self, leveldb_dir=leveldb_dir):
        print('__init__ NGramChecker')
        self.pinyin_db = plyvel.DB(os.path.join(leveldb_dir, 'py_db'), create_if_missing=False)
        self.freq_db = plyvel.DB(os.path.join(leveldb_dir, 'word_freq_db'), lru_cache_size=1024*1024*10)
        freq_size = 0
        max_freq = 0
        total_freq = 0
        with self.freq_db.iterator() as it:
            for k, v in it:
                v = int(v)
                if v > max_freq:
                    max_freq = v
                freq_size += 1
                total_freq += v
        self.freq_size = freq_size
        self.max_freq = max_freq
        self.MIN_UNION_WORD_FREQ_PROB = math.log(1.0 / (max_freq + freq_size)) * 3.0
        self.MIN_WORD_FREQ_PROB = math.log(1.0 / total_freq) - 1
        self.MIN_PROB = -1.0 * sys.maxsize
        self.TOTAL_WORDS = total_freq
        self._candidate_gen = CandidateGen(self.pinyin_db, self.freq_db)
        self.tokenizer = Tokenizer()
        self.tokenizer.initialize()  # for loading model
        posseg.initialize()
        print('__init__ NGramChecker done')

    @lru_cache(maxsize=int(os.environ.get("NGram_CalcProb_CacheSize", "2000")))
    def calc_prob(self, a, b, weight):
        if not a:
            if not b:
                raise Exception('pre word and next word should not be None together')
            return self.MIN_UNION_WORD_FREQ_PROB

        # pruning
        if self.freq_db.get(a.encode("utf-8"), None) is None or self.freq_db.get((a + b).encode("utf-8"), None) is None:
            return self.MIN_UNION_WORD_FREQ_PROB / weight

        score = math.log((1.0 * int(self.freq_db.get((a + b).encode('utf-8'), 0)) + 1) /
                         (int(self.freq_db.get(a.encode('utf-8'), 0)) + self.freq_size)) / weight
        return score

    @lru_cache(maxsize=100)
    def origin_factor(self, length):
        '''
        给原始句子乘上对应长度的放大系数
        '''
        if length <= 3:
            return -0.125 * (length - 3) + 0.5
        else:
            return 0.5

    def check_sentence(self, sentence, max_back_grams=3, raw_only=False, word_seg_list=None):
        if not word_seg_list: word_seg_list = []
        candidate_list = CandidateList()
        for curr_right in range(len(sentence)):
            max_offset = min(curr_right, max_back_grams)
            for curr_left in range(curr_right, curr_right - max_offset - 1, -1):
                original_word = sentence[curr_left:curr_right+1]
                is_raw_only = raw_only
                for (cand_word, cand_weight) in self._candidate_gen.gen_candidates(original_word, is_raw_only).items():
                    if '么' in cand_word:
                        print(cand_word, cand_weight)
                    if len(cand_word) != len(original_word): continue
                    candidate = Candidate(cand_word, cand_word == original_word, None, self.MIN_PROB)
                    for prev_candidate in candidate_list[curr_left-1]:
                        proba = self.calc_prob(prev_candidate.word, candidate.word, cand_weight)
                        if prev_candidate.is_flag and not candidate.is_original and proba == self.MIN_UNION_WORD_FREQ_PROB: continue
                        if abs(proba - candidate.proba) < 1e-7 and not (candidate.is_original and prev_candidate.is_original): continue
                        if prev_candidate.is_flag and candidate.word in word_seg_list and candidate.is_original and (abs(proba - self.MIN_UNION_WORD_FREQ_PROB) < 1e-7):
                            proba = self.MIN_UNION_WORD_FREQ_PROB * self.origin_factor(len(candidate.word))
                        if prev_candidate.word in word_seg_list and candidate.is_original and prev_candidate.is_original and (abs(proba - self.MIN_UNION_WORD_FREQ_PROB) < 1e-7):
                            proba = self.MIN_UNION_WORD_FREQ_PROB * self.origin_factor(len(prev_candidate.word))
                        if curr_right == len(sentence) - 1: proba += self.calc_prob(candidate.word, "E", cand_weight)
                        proba += prev_candidate.proba
                        if proba > candidate.proba or (abs(proba - candidate.proba) < 1e-7 and prev_candidate.is_original):
                            candidate.proba = proba
                            candidate.prev_candidate = prev_candidate
                    if not candidate.prev_candidate: continue
                    candidate_list[curr_right].append(candidate)
                    if candidate.is_original: candidate_list.origin_sequence = candidate
        # 找最优路径
        max_prob = self.MIN_PROB
        max_candidate = None
        for candidate in candidate_list[len(sentence) - 1]:
            if candidate.proba > max_prob:
                max_prob = candidate.proba
                max_candidate = candidate
        new_words = []
        error_index = []
        line = str(sentence)
        while max_candidate.is_flag != True:
            candidate_words = max_candidate.word
            if line[-len(candidate_words):] != candidate_words:
                error_index.append((len(line) - len(candidate_words), len(line) - 1))
            line = line[:-len(candidate_words)]
            new_words.append(candidate_words)
            max_candidate = max_candidate.prev_candidate
        new_words.reverse()
        error_index.reverse()
        origin_prob = candidate_list.origin_sequence.proba
        error_word_index = []
        curr_idx = 0
        for word_idx, word in enumerate(new_words):
            if (curr_idx, curr_idx + len(word) - 1) in error_index:
                error_word_index.append(word_idx)
            curr_idx += len(word)
        check_passed = False
        for error_idx, word_idx in enumerate(error_word_index):
            orig_word = sentence[error_index[error_idx][0]:error_index[error_idx][1]+1]
            if word_idx - 1 >= 0:
                if self.calc_prob(new_words[word_idx - 1], new_words[word_idx], 1) > self.calc_prob(
                        new_words[word_idx - 1], orig_word, 1):
                    check_passed = True
            if word_idx + 1 < len(new_words):
                if self.calc_prob(new_words[word_idx], new_words[word_idx + 1], 1) > self.calc_prob(
                        orig_word, new_words[word_idx + 1], 1):
                    check_passed = True
        if (max_prob / len(sentence) > float(os.getenv("NGramThreshold", -12.0))) and (
                ((max_prob - origin_prob) / -origin_prob > 0.06) or check_passed):
            return ''.join(new_words), error_index, max_prob / len(sentence)
        else:
            return '', [], max_prob / len(sentence)

    def check_sentence_with_word_seg(self, sentence):
        """词查错"""
        cut_list = self.tokenizer.lcut(sentence)
        results = []
        curr_idx = len(cut_list[0])
        for word_idx in range(1, len(cut_list)):
            orig_word = cut_list[word_idx]
            possible_replacements = []
            confidence = []
            for candidate in self._candidate_gen.gen_candidates(cut_list[word_idx], raw_only=False):
                if candidate == orig_word: continue
                proba_passed = False
                if word_idx - 1 >= 0:
                    orig_proba = self.calc_prob(cut_list[word_idx - 1], orig_word, 1)
                    new_proba = self.calc_prob(cut_list[word_idx - 1], candidate, 1)
                    if orig_proba > 0.3 * self.MIN_UNION_WORD_FREQ_PROB: proba_passed = True
                    if new_proba > orig_proba and orig_proba <= 0.8 * self.MIN_UNION_WORD_FREQ_PROB and not proba_passed:
                        if candidate in possible_replacements: continue
                        old_cut_result = posseg.lcut(f"{cut_list[word_idx - 1]}{orig_word}")
                        new_cut_result = posseg.lcut(f"{cut_list[word_idx - 1]}{candidate}")
                        if len(old_cut_result) != len(new_cut_result): continue
                        if [pos_tag for _, pos_tag in old_cut_result] != [pos_tag for _, pos_tag in new_cut_result]: continue
                        possible_replacements.append(candidate)
                        confidence.append(self.calc_prob(cut_list[word_idx - 1], candidate, 1))
                if word_idx + 1 < len(cut_list):
                    orig_proba = self.calc_prob(orig_word, cut_list[word_idx + 1], 1)
                    new_proba = self.calc_prob(candidate, cut_list[word_idx + 1], 1)
                    if orig_proba > 0.3 * self.MIN_UNION_WORD_FREQ_PROB: proba_passed = True
                    if new_proba > orig_proba and orig_proba <= 0.8 * self.MIN_UNION_WORD_FREQ_PROB and not proba_passed:
                        if candidate in possible_replacements: continue
                        old_cut_result = posseg.lcut(f"{orig_word}{cut_list[word_idx + 1]}")
                        new_cut_result = posseg.lcut(f"{candidate}{cut_list[word_idx + 1]}")
                        if len(old_cut_result) != len(new_cut_result): continue
                        if [pos_tag for _, pos_tag in old_cut_result] != [pos_tag for _, pos_tag in new_cut_result]: continue
                        possible_replacements.append(candidate)
                        confidence.append(self.calc_prob(candidate, cut_list[word_idx + 1], 1))

            if len(possible_replacements) > 0:
                index_s = sentence.find(orig_word)
                results.append([''.join(possible_replacements[0]), (index_s, index_s+len(orig_word)-1), max(confidence)])
            curr_idx += len(orig_word)
        if len(results) == 0: return []
        max_confidence = -9999
        max_confidence_result = None
        for result in results:
            if result[-1] <= max_confidence: continue
            max_confidence = result[-1]
            max_confidence_result = result
        return [max_confidence_result]

    def main(self, cur_text):

        sentences_info = preprocess_and_split_sentences(cur_text)
        texts = []
        mappings = []
        for sentence, mapping in sentences_info:
            print(sentence)
            texts.append(sentence)
            char_map = []
            for idx, char in enumerate(sentence):
                map_start, map_end = mapping[idx]
                if re.match("[A-Z]", char):
                    char_map.append(map_end)
                else:
                    char_map.append(map_start)
            mappings.append(char_map)

        results = []
        error_indexes = set()
        for text_id, text in enumerate(texts):
            raw_cut_list = self.tokenizer.lcut(text, HMM=False)
            correction, ec_index, avg_prob = self.check_sentence(sentence=text, word_seg_list=raw_cut_list)
            corrected_sentence = "".join(correction)
            if distance(text, corrected_sentence) > 3:
                continue
            for s, e in ec_index:
                _s, _e = mappings[text_id][s], mappings[text_id][e]
                results.append(["".join(correction[s: e + 1]), (_s, _e), avg_prob])
                error_indexes.add((_s, _e))
        for text_id, text in enumerate(texts):
            result_seg = self.check_sentence_with_word_seg(text)
            for correction, ec_index, avg_prob in result_seg:
                ec_index = (mappings[text_id][ec_index[0]], mappings[text_id][ec_index[1]])
                if ec_index in error_indexes:
                    continue
                results.append([correction, ec_index, avg_prob])

        return [{"error_correction": [cur[0]], "origin": cur_text[cur[1][0]:cur[1][1]+1],
                 "start_offset": cur[1][0], "end_offset": cur[1][1]}for cur in results]


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


if __name__ == '__main__':
    _text = "那是我的侄子和侄女的招片。"
    # _text = "上银行，使用网上银行可以在网上转帐，也可以在网上购买东西。"
    # _text = "我中标得到那几本课本后，他就联系我怎幺寄送。"
    _text = "他就联系我怎幺寄送。"
    _text = "所以我们早上买午饭去工作或做弁当拿去工作"
    _text = "两个月之前，我在韩国的时候，用英语说比用汉语说得跟容易。"
    checker = NgramCorrector()
    _results = checker.main(_text)
    print(_results)
