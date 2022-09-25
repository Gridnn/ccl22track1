# coding: utf-8
import pandas as pd
import numpy as np
from ngram_corrector import NgramCorrector
from post_mlm import MLMProcessor

ngram_processor = NgramCorrector()
mlm_processor = MLMProcessor()


def predict(text):
    ngram_results = ngram_processor.main(text)
    if not ngram_results:
        return ngram_results
    mlm_results = mlm_processor.predict(text, ngram_results)
    final_result = list(filter(lambda x: x['status'], mlm_results))
    return final_result


def batch_predict():
    data_path = "./data/track1/test/yaclc-csc_test.src"
    save_path = "result.xlsx"
    texts_ids = [cur.split('\t') for cur in open(data_path, 'r', encoding='utf-8').read().split('\n') if '\t' in cur]
    raw_index = list(np.array(texts_ids)[:, 0])
    text_list = list(np.array(texts_ids)[:, 1])
    result_list = list(map(predict, text_list))
    final_result = []
    for index, text, result in zip(raw_index, text_list, result_list):
        if not result:
            cur_item = [index, text]
            cur_item.extend(['']*5)
            final_result.append(cur_item)
            continue
        for cur in result:
            cur_item = [index, text]
            cur_item.extend([v for k, v in cur.items()])
            final_result.append(cur_item)

    print(final_result)

    # columns = ['index', 'text', 'error_correction', 'origin', 'start_offset', 'end_offset', 'status']
    # pd.DataFrame(final_result, columns=columns).to_excel(save_path, encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    # text = "那是我的侄子和侄女的招片。"
    # text = "在一个银行工作，现在他还在那里攻作。"
    # text = "我喜欢看它可爱的圆圆地眼睛。"
    # text = "我的MP3在我穿大衣的口带里。"
    # _text = "上银行，使用网上银行可以在网上转帐，也可以在网上购买东西。"
    # predict(_text)
    batch_predict()

