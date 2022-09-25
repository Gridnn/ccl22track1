# coding: utf-8
import os
os.environ['TF_KERAS'] = '1'
import tensorflow as tf
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array, sequence_padding
from Levenshtein import editops
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


class MLMProcessor:
    def __init__(self):
        self.maxlen = 512
        self.MODEL_DIR = "./model/chinese_L-12_H-768_A-12"
        self.load()
        self.threshold = 0.08
        self.batch_size = 32

    def load(self):

        config_path = os.path.join(self.MODEL_DIR, 'bert_config.json')
        checkpoint_path = os.path.join(self.MODEL_DIR, 'bert_model.ckpt')
        dict_path = os.path.join(self.MODEL_DIR, 'vocab.txt')

        self.tokenizer = Tokenizer(dict_path, do_lower_case=False)  # 建立分词器
        self.model = build_transformer_model(
            config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
        )  # 建立模型，加载权get_重

    def load_data(self, text, results):
        """
        将数据整理为batch的格式
        :param input_data:
        :return: batch_tokens, batch_segments, 对应的corrected_items
        """
        batch_tokens = []
        batch_segments = []
        corrected_items = []
        # results
        for result_idx, result in enumerate(results):
            start_offset = result["start_offset"]
            end_offset = result["end_offset"]

            tmp_cor_items = []
            for text_after in result['error_correction']:
                text_before = result['origin']
                input_text = list(text_before)
                cor_items = []
                if len(text_after) == len(text_before):
                    for i in range(len(text_after)):
                        if text_after[i] == text_before[i]:
                            continue
                        cor_items.append((start_offset + i, text_after[i]))
                        input_text[i] = '[MASK]'

                input_text = list(text[:start_offset]) + input_text + list(text[end_offset+1:])
                assert len(input_text) == len(text)
                token_ids, segment_ids = self.tokenizer.encode(['CLS'] + input_text + ['SEP'],
                                                               maxlen=self.maxlen)
                batch_tokens.append(token_ids)
                batch_segments.append(segment_ids)

                tmp_cor_items.append(cor_items)
            corrected_items.append(tmp_cor_items)
            if len(batch_tokens) >= self.batch_size:
                batch_tokens = sequence_padding(batch_tokens)
                batch_segments = sequence_padding(batch_segments)

                yield batch_tokens, batch_segments, corrected_items
                batch_tokens, batch_segments, corrected_items = [], [], []

        batch_tokens = sequence_padding(batch_tokens)
        batch_segments = sequence_padding(batch_segments)
        yield batch_tokens, batch_segments, corrected_items

    def predict(self, sentence_lst, results):
        # 利用预训练模型判断当前的结果是否接受
        mlm_results = []
        for batch_data in self.load_data(sentence_lst, results).__iter__():
            model_preds = self.model.predict(to_array(batch_data[0], batch_data[1]))
            # 对当前batch的模型输出进行处理
            # 确定result在model outputs中的位置
            input_offset = -1
            for result_idx, result_item in enumerate(batch_data[2]):
                # 一个纠错结果对应多个候选集的情况, 当前的判断结果对应mlm_results
                passed_corrections = []
                for item in result_item:
                    input_offset += 1
                    item_passed = True
                    for offset, can_char in item:
                        judgement = model_preds[input_offset][offset + 1][self.tokenizer.token_to_id(can_char)] <= self.threshold
                        if judgement:
                            item_passed = False
                            break
                    if item_passed:
                        passed_corrections.append(results[result_idx]['error_correction'])
                if len(passed_corrections) == 0:
                    mlm_results.append(False)
                else:
                    mlm_results.append(True)
        final_result = []
        for idx in range(len(results)):
            row = results[idx]
            row['status'] = mlm_results[idx]
            final_result.append(row)
        return final_result


# if __name__ == '__main__':
#     processor = MLMProcessor()
#
#     text = "上银行，使用网上银行可以在网上转帐，也可以在网上购买东西。"
#
#     results = [{'error_correction': ['转账'], 'origin': '转帐', 'start_offset': 15, 'end_offset': 16}]
#
#     input_data = {"text": text, "results": results}
#     print(input_data)
#     mlm_results = processor.predict(text, results)
#     print(mlm_results)
