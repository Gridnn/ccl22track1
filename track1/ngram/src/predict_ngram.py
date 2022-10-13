# coding: utf-8
import pandas as pd
import numpy as np
from src.post_mlm import MLMProcessor
from src.ngram_corrector import config, NgramCorrector

mlm_processor = MLMProcessor()
ngram_processor = NgramCorrector(config)


def predict_ngram(text):
    ngram_results = ngram_processor.correct(text)
    if not ngram_results:
        print(ngram_results)
        return ngram_results
    mlm_results = mlm_processor.predict(text, ngram_results)
    final_result = list(filter(lambda x: x['status'], mlm_results))
    return final_result


def save_txt(txt_path, data):
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))


def batch_predict():
    data_path = "../data/track1/test/yaclc-csc_test.src"
    save_path = "track1_result.xlsx"
    texts_ids = [cur.split('\t') for cur in open(data_path, 'r', encoding='utf-8').read().split('\n') if '\t' in cur]
    raw_index = list(np.array(texts_ids)[:, 0])
    text_list = list(np.array(texts_ids)[:, 1])

    result_list = list(map(predict_ngram, text_list))
    final_result = []
    txt_list = []
    for index, text, result in zip(raw_index, text_list, result_list):
        if not result:
            cur_item = [index, text]
            cur_item.extend(['']*5)
            final_result.append(cur_item)
            txt_list.append('\t'.join([text, text]))
            continue
        cur_text = ''.join(list(text))
        for cur in result:
            cur_item = [index, text]
            cur_item.extend([v for k, v in cur.items()])
            cur_text = cur_text.replace(cur['origin'], cur['error_correction'][0])
            final_result.append(cur_item)
        txt_list.append('\t'.join([text, cur_text]))

    columns = ['index', 'text', 'error_correction', 'origin', 'start_offset', 'end_offset', 'status']
    pd.DataFrame(final_result, columns=columns).to_excel(save_path, encoding='utf_8_sig', index=False)
    save_txt(data_path.replace('.src', '.txt'), txt_list)


if __name__ == '__main__':
    _text = "满不满意？我听说中国化了很多很多钱，所以我想知道到底值不值得。"
    # predict_ngram(_text)

    batch_predict()

