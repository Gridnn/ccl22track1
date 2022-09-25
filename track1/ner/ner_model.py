# coding: utf-8

import os
from typing import Union
os.environ['TF_KERAS'] = '1'

from bert4keras.backend import K, keras
from bert4keras.layers import ConditionalRandomField
from bert4keras.snippets import ViterbiDecoder, sequence_padding, DataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model

import json
import re

from copy import deepcopy
from functools import partial
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam

import shutil

from bert4keras.models import build_transformer_model
from collections import namedtuple


if len(tf.config.list_physical_devices("GPU")) == 0:
    num_cpu = os.cpu_count()
    max_threads = int(os.getenv("NUM_CPU", 0.5*num_cpu))
    if max_threads <= 0 or max_threads > num_cpu:
        max_threads = int(0.5*num_cpu)
    tf.config.threading.set_inter_op_parallelism_threads(max_threads)
    tf.config.threading.set_intra_op_parallelism_threads(max_threads)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


class LMConfig(object):
    def __init__(self):

        # embedding
        self.max_len = 128

        # model params
        self.pretrain_model = 'bert'
        self.bert_layers = 12
        self.freeze_layers = 8

        # optimizer
        self.learning_rate = 3e-5     # bert_layers越小，学习率应该要越大

        # train process
        self.epochs = 20
        self.batch_size = 32
        self.early_stopping_patience = 5
        self.labels = []
        self.val_size = 0.2
        self.model_selection_metrics = ["f1"]
        self.min_delta = [0]
        self.random_state = 2022

        # path
        self.test_data_path = None
        self.config_path = None
        self.checkpoint_path = None
        self.dict_path = None
        self.label_path = './label.txt'
        self.model_path = "./model"

        # crf only
        self.crf_lr_multiplier = 200  # 扩大CRF层的学习率

    def set_para(self, para_dict: dict):
        """
        设置某些para的值
        :param para_dict: 需要更改的配置的键值对
        :type para_dict: dict
        :return: 返回未存在的参数
        :rtype: ''
        """
        untracked_para = []
        for key, value in para_dict.items():
            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                untracked_para.append(key)
        return untracked_para
    
    def get_para(self):
        """
        获取模型参数配置，以Dict返回

        """
        return self.__dict__

def tag_to_entity_list(string, tags):
    """
    BIO标注的实体抽取方法，不能用于单个实体用"S"表示的标注方式
    :param string: 需注意字符的长度和标签的长度
    :param tags:
    :return:
    """
    item = []
    idx = 0

    while idx < len(string):
        if tags[idx][0] == 'B':
            entity_value = string[idx]
            current_entity = tags[idx][2:]
            entity_start = idx
            while idx < len(string):
                if idx + 1 < len(string) and tags[idx+1][0] != 'B' and tags[idx+1][2:] == current_entity:
                    idx += 1
                    entity_value += string[idx]
                else:
                    idx += 1
                    item.append({"value": entity_value, "start": entity_start,
                                 "end": idx, "tag": current_entity})
                    break
        else:
            idx += 1
    return item

def entity_tags_to_bio(tags, text_list):
    """
        Entity List -> BIO
    :param tags: list of entity tags, [[{"start":0, "end":4, "value":"达观数据", "tag":"ORG"}], ...]
    :param text_list: list of origin text
    :return: BIO序列
    """
    bio_list = []
    for tag, text in zip(tags, text_list):
        bio = ['O'] * len(text)
        for item in sorted(tag, key=lambda x: x['start']):
            start, end, label = item['start'], item['end'], item['tag']
            if set(bio[start: end]) - {'O'}:
                continue
            bio[start] = 'B-{}'.format(label)
            if end - start > 1:
                bio[start + 1: end] = ['I-{}'.format(label)] * (end - start - 1)
        bio_list.append(bio)
    return bio_list

def filter_white_sample(data: list):
    """
    过滤白样本，要求白样本在data中是[[], [], len(sentence), text_index]
    """
    data_filter = filter(lambda x: x[0], data)
    return [[sample[0], sample[1]] for sample in data_filter]

SENTENCE_SPLIT_PATTERN = re.compile(r'(\n|。)')

def cut_sentence(passage, max_len=256):
    max_len = max_len - 2  # 2 for additional [CLS] & [SEP] tag
    matches = SENTENCE_SPLIT_PATTERN.split(passage)
    values = matches[::2]
    delimiters = matches[1::2] + ['']
    merged_sentences = []
    merged_sentence = ''
    for v, d in zip(values, delimiters):
        if len(merged_sentence) + len(v) + len(d) <= max_len:
            merged_sentence += (v + d)
        else:
            merged_sentences.append(merged_sentence)
            merged_sentence = v + d
            if len(merged_sentence) > max_len:
                # 超过max_len则直接截断为多个句子，暂不考虑边界case
                i = 0
                while i < len(merged_sentence):
                    merged_sentences.append(merged_sentence[i:i+max_len])
                    i += max_len
                merged_sentence = ''
    if merged_sentence:
        merged_sentences.append(merged_sentence)
    return merged_sentences

PretrainInfo = namedtuple('PretrainInfo', ['model_dir', 'builtin_models'])

def get_pretrain_info(model_dir=None):
    """
    输入model_dir（或者None）
    返回: 
    1.model_dir（当输入None时返回系统默认路径）;
    2.预训练模型信息
    Params:
        model_dir: 统一的预训练模型存放路径
    """
    if model_dir is None:
        model_dir = os.getenv('PRETRAIN_MODEL_DIR',
            os.path.join(os.path.expanduser('~'), '.dgmodel', 'models')
        )
    os.makedirs(model_dir, exist_ok=True)

    # 优先读取指定model_dir下的配置文件，其次是包中内置的文件
    builtin_path = os.path.join(model_dir, 'models.json')
    if not os.path.exists(builtin_path):
        builtin_path = os.path.join(os.path.dirname(__file__), 'models.json')
    with open(builtin_path, encoding='utf-8') as f:
        builtin_models = json.load(f)

    return PretrainInfo(model_dir, builtin_models)

class PretrainLanguageModel(object):
    """预训练模型统一管理（bert4keras）
    """

    def __init__(self,
                 config: Union[dict, LMConfig],
                 model_dir: str = None) -> None:
        if isinstance(config, LMConfig):
            config = config.get_para()
        self.checkpoint_path = None
        self.config_path = None
        self.dict_path = None
        self.model_type = 'bert'
        self.bert_layers = 12
        self.freeze_layers = 0
        pretrain_info = get_pretrain_info(model_dir)
        self.model_dir, self.builtin_models = pretrain_info.model_dir, pretrain_info.builtin_models

        if config["checkpoint_path"] is not None and config["config_path"] is not None and config["dict_path"] is not None:
            local_config_path = os.path.join(config["model_path"], os.path.split(config["config_path"])[1])
            local_dict_path = os.path.join(config["model_path"], os.path.split(config["dict_path"])[1])
            if os.path.exists(local_config_path) and os.path.exists(local_dict_path):
                config["config_path"] = local_config_path
                config["dict_path"] = local_dict_path
                self.config_path = local_config_path
                self.dict_path = local_dict_path
                self._init_build_params(config)

        if self.config_path is None or self.dict_path is None:
            self._init_params(config)

        self.config = config

    @staticmethod
    def _check_exists(config):
        for p in ['checkpoint_path', 'config_path', 'dict_path']:
            path = config[p]
            if config[p] is None:
                return False
            if path.endswith('.ckpt'):
                path += '.index'
            if not os.path.exists(path):
                return False
        return True

    def _init_build_params(self, config: dict):
        model_name = config.get('pretrain_model', 'ernie')   # 默认为ernie
        self.model_type = self.builtin_models[model_name]['type']    # bert4keras内部的模型类型
        # 读取模型配置中的总层数，自定义加载层数和锁定层数必须小于总层数
        with open(self.config_path, encoding='utf-8') as f:
            model_conf = json.load(f)
        if config.get('bert_layers') and \
            config['bert_layers'] < model_conf['num_hidden_layers']:
            self.bert_layers = config['bert_layers']
        else:
            self.bert_layers = model_conf['num_hidden_layers']
        self.freeze_layers = min(config.get('freeze_layers', 0), self.bert_layers)

    def _init_params(self, config: dict) -> None:
        model_name = config.get('pretrain_model', 'ernie')   # 默认为ernie
        if model_name not in self.builtin_models:
            raise ValueError(f'{model_name} is not a built-in model.')
        model_info = self.builtin_models[model_name]

        # 1）首先检查config中是否存在自定义的模型路径
        if not self._check_exists(config):
            save_dir = os.path.join(self.model_dir, model_info['name'])
            os.makedirs(save_dir, exist_ok=True)
            # 2）再次检查本地模型库中是否存在指定模型
            config['checkpoint_path'] = os.path.join(save_dir, model_info['checkpoint_file'])
            config['config_path'] = os.path.join(save_dir, model_info['config_file'])
            config['dict_path'] = os.path.join(save_dir, model_info['dict_file'])
            if not self._check_exists(config):
                raise FileNotFoundError('Fail to find pretrain checkpoints in expected directory')

        self.checkpoint_path = config['checkpoint_path']
        self.config_path = config['config_path']
        self.dict_path = config['dict_path']

        self._init_build_params(config)

    def build(self):
        model = build_transformer_model(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model=self.model_type,
            num_hidden_layers=self.bert_layers,
            return_keras_model=False
        )
        return self.freeze_model_layers(model)

    def freeze_model_layers(self, model):
        """锁定部分transformer层不参与训练，同时锁定Embedding层
        """
        if self.freeze_layers > 0:
            for layer in model.model.layers:
                if 'Embedding' in layer.name:
                    layer.trainable = False
                for i in range(int(self.freeze_layers)):
                    if 'Transformer-%s-' % i in layer.name:
                        layer.trainable = False
        return model

    def save(self):
        """复制预训练模型config和vocab文件到指定目录，并返回存储路径
        """
        target_config_path = os.path.join(self.config["model_path"], os.path.split(self.config_path)[1])
        if self.config_path != target_config_path:
            shutil.copy(self.config_path, self.config["model_path"])
        target_dict_path = os.path.join(self.config["model_path"], os.path.split(self.dict_path)[1])
        if self.dict_path != target_dict_path:
            shutil.copy(self.dict_path, self.config["model_path"])
        return target_config_path, target_dict_path

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_X, batch_y = [], []
        for is_end, (X, y) in self.sample(random):
            batch_X.append(X)
            batch_y.append(y)
            if len(batch_X) == self.batch_size or is_end:
                batch_X = list(map(lambda x: sequence_padding(x), zip(*batch_X)))
                batch_y = list(map(lambda x: sequence_padding(x,
                                                              seq_dims=len(x[0].shape) if isinstance(x[0], np.ndarray) else 1),
                                   zip(*batch_y)))
                if len(batch_X) == 1:
                    batch_X = batch_X[0]
                if len(batch_y) == 1:
                    batch_y = batch_y[0]
                yield batch_X, batch_y
                batch_X, batch_y = [], []

class Evaluator(keras.callbacks.Callback):
    """评估与保存(EarlyStopping)
    """
    def __init__(self, eval_func, eval_data, save_func, patience,
                 metrics=["f1"], min_delta=[0], verbose=1):
        super().__init__()
        self.metrics = metrics
        self.min_delta = min_delta
        self.best_val_metrics = self._init_best_metrics()
        self.best_epoch = 0
        self.eval_func = eval_func
        self.eval_data = eval_data
        self.save_func = save_func
        self.patience = patience or np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.best_val_metrics = self._init_best_metrics()
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Stop training when either validation metric has stopped improving.
        """
        is_best = True
        temp_metrics = []
        for i, metric in enumerate(self.metrics):
            if "f1" in metric:
                val_f1 = self.eval_func(self.eval_data)
                temp_metrics.append(val_f1)
                is_best = is_best and (val_f1 - self.min_delta[i] > self.best_val_metrics[i])
            elif "loss" in metric:
                val_loss = logs.get("val_loss")
                temp_metrics.append(val_loss)
                is_best = is_best and (val_loss + self.min_delta[i] < self.best_val_metrics[i])

        if is_best:
            self.best_val_metrics = deepcopy(temp_metrics)
            self.best_epoch = epoch + 1
            self.wait = 0
            self.save_func()
        else:
            self.wait += 1
        if self.verbose > 0:
            message = ', '.join([f"val_{metric}: {metric_value}"
                                 for metric, metric_value in zip(self.metrics, temp_metrics)])
            message += '\t'+', '.join([f"best_{metric}: {metric_value}"
                                 for metric, metric_value in zip(self.metrics, self.best_val_metrics)])
            message += '\t'+f'best epoch: {self.best_epoch}'
            print(f"\nEpoch {epoch+1}:\t{message}")

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose > 0:
                print(f"Restoring model from the end of the best epoch: Epoch {self.best_epoch}.")
            self.model.stop_training = True
            
    def on_train_end(self, logs=None):
        if self.best_epoch == 0:
            self.save_func()
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch+1}: Early Stopping.")

    def _init_best_metrics(self):
        best_val_metrics = []
        for metric in self.metrics:
            if "f1" in metric:
                best_val_metrics.append(-np.inf)
            elif "loss" in metric:
                best_val_metrics.append(np.inf)
            else:
                raise NotImplementedError(f"metric {metric} is currently not supported in LMModel")
        return best_val_metrics

class LMModel(object):
    """采用BI方式进行序列标注训练"""

    def __init__(self, config: Union[dict, LMConfig], logger=None):
        if isinstance(config, dict):
            lm_config = LMConfig()
            lm_config.set_para(config)
            config = lm_config
        self.config = config
        self.model = None
        self.logger = logger
        # 建立分词器
        self.pretrain_model = PretrainLanguageModel(config)
        self.tokenizer = Tokenizer(self.pretrain_model.dict_path, do_lower_case=True)
        # 类别映射: id2label: list; label2id: dict
        self.id2label = None
        self.label2id = None
        self.num_labels = 0
        self.build_label_from_config()
        self.custom_objects = None
        os.makedirs(self.config.model_path, exist_ok=True)

        self.CRF = None
        self.viterbi_decoder = None
        self._custom_objects = {
            "sparse_accuracy": ConditionalRandomField.sparse_accuracy
        }

    def prepare_train(self, train_text_df):
        # build label
        self.build_label(train_text_df)
        # train test split
        df_train = train_text_df
        df_val = pd.read_csv(self.config.test_data_path, dtype='object',
                                encoding='utf-8', usecols=['text', 'label']).replace(
                                    {"": None}).dropna(subset=['label']).fillna(
                                        "").reset_index(drop=True)

        df_train = df_train.fillna('')
        ix1 = (df_train['text']!='') & (df_train['label']!='')
        df_train = df_train.loc[ix1].reset_index(drop=True)
        df_val = df_val.fillna('')
        ix1 = (df_val['text']!='') & (df_val['label']!='')
        df_val = df_val.loc[ix1].reset_index(drop=True)

        # save pretain config and vocab
        config_path, dict_path = self.pretrain_model.save()
        self.config.set_para({"config_path": config_path, "dict_path": dict_path})
        # save config parameters
        with open(os.path.join(self.config.model_path, 'config.json'), 'w') as f:
            f.write(json.dumps(self.get_config(), ensure_ascii=False, indent=2))

        return df_train, df_val

    def train(self, train_text_df, dev_text_df=None):
        print("train from scratch.")
        df_train, df_val = self.prepare_train(train_text_df)
        train_data = self.transform_data(df_train)
        train_data = filter_white_sample(train_data)
        train_generator = data_generator(train_data, batch_size=self.config.batch_size)
        val_data = self.transform_data(df_val)
        val_data = filter_white_sample(val_data)
        _val_generator = data_generator(val_data, batch_size=len(val_data))
        val_data = [(_val_X, _val_y) for _val_X, _val_y in _val_generator][0]
        # build model
        if self.model: del self.model
        self.build_model()
        self.model = self.compile_model(self.define_loss(), self.define_optimizer(), self.define_metrics())
        evaluator = Evaluator(self.evaluate, df_val, partial(self.save_model, model_save=True),
                              self.config.early_stopping_patience, self.config.model_selection_metrics,
                              self.config.min_delta)
        self.model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=self.config.epochs,
            callbacks=[evaluator],
            validation_data=val_data,
            validation_batch_size=self.config.batch_size
        )
        # 载入best模型
        del self.model
        self.load_model()

    def suppress_impossible(self, trans):
        for i, row in enumerate(trans):
            head_label = self.id2label[i]
            for j in range(len(row)):
                tail_label = self.id2label[j]
                # 例如：I-时间 ——> I-地点
                if i != j and head_label[0] == 'I' and tail_label[0] == 'I':
                    row[j] = -1e12
                # 例如: O ——> I-时间
                if head_label[0] == 'O' and tail_label[0] == 'I':
                    row[j] = -1e12
                # 例如：B-时间 ——> I-地点
                if head_label[0] == 'B' and tail_label[0] == 'I' and head_label[1:] != tail_label[1:]:
                    row[j] = -1e12
        return trans

    def evaluate(self, data, metrics="f1"):
        """评测函数
        """
        # crf每次eval需要更新一下trans参数
        trans = K.eval(self.CRF.trans)
        # 这一步可视情况添加
        trans = self.suppress_impossible(trans)
        self.viterbi_decoder.trans = trans

        X, Y, Z = 1e-10, 1e-10, 1e-10
        Ts, Rs = [], []
        data['label'] = data['label'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        predicts = self.predict_batch(data['text'].tolist())

        for i, (y_true, y_pred) in enumerate(zip(data['label'], predicts)):
            Ts.append([(item['value'], item['tag'], item['start']) for item in y_true])
            Rs.append([(item['value'], item['tag'], item['start']) for item in y_pred])
            R = set(Rs[i])
            T = set(Ts[i])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
        metric_dic = {
            'f1': 2 * X / (Y + Z),
            'precision': X / Y,
            'recall': X / Z
        }
        return metric_dic[metrics]

    def build_label_from_config(self):
        # 类别映射
        if not self.config.labels:
            # 读取本地已保存的config文件
            config_file_path = os.path.join(self.config.model_path, 'config.json')
            if os.path.exists(config_file_path):
                with open(config_file_path) as f:
                    config = json.load(f)
                self.config.set_para({"labels": config["labels"]})
        if not self.config.labels:
            self.id2label = []
        else:
            self.id2label = ['O']
            for label in self.config.labels:
                for tag in ['B', 'I']:
                    self.id2label.append('{}-{}'.format(tag, label))
        self.label2id = {l: i for i, l in enumerate(self.id2label)}
        self.num_labels = len(self.id2label)

    def build_label(self, train_text_df):
        if not self.config.labels:
            train_text_df['label'] = train_text_df['label'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            label_lst = train_text_df['label'].apply(lambda x: [tag["tag"] for tag in x]).sum()
            self.config.set_para({"labels": sorted(set(label_lst))})
        self.build_label_from_config()

    def build_model(self):
        # create bert and load pre-trained parameters on demand
        bert = self.pretrain_model.build()
        output = Dense(self.num_labels, dtype='float32', kernel_initializer=bert.initializer)(bert.model.output)
        self.CRF = ConditionalRandomField(lr_multiplier=self.config.crf_lr_multiplier, dtype='float32')
        output = self.CRF(output)
        model = Model(bert.model.input, output)
        self.model = model
        print("final model summary")
        self.model.summary()
        trans = K.eval(self.CRF.trans)
        self.viterbi_decoder = ViterbiDecoder(trans=trans)

    def define_loss(self):
        return self.CRF.sparse_loss

    def define_optimizer(self):
        optimizer_params = {
            'learning_rate': self.config.learning_rate,
        }
        optimizer = Adam
        optimizer = optimizer(**optimizer_params)
        return optimizer

    def define_metrics(self):
        return

    def compile_model(self, loss, optimizer, metrics, loss_weights=None, model=None):
        if model is None:
            model = self.model
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics,
                      loss_weights=loss_weights)
        return model

    def transform_data(self, df):
        """转换带标签的数据"""
        data = [] # [[X:list, y: list, len_text, text_index]]

        def _add_data(sentence_lst, labels_lst, text_index):
            assert len(sentence_lst) == len(labels_lst)
            for sentence, labels in zip(sentence_lst, labels_lst):
                assert len(sentence) == len(labels)
                # 如果序列的标注只有一种，则标注置空，需在训练前过滤
                if len(set(labels)) == 1:
                    data.append([[], [], len(sentence), text_index])
                else:
                    # [CLS]token
                    token_ids = [self.tokenizer.token_to_id(self.tokenizer._token_start)]
                    label_ids = [self.label2id["O"]]
                    # word token
                    token_ids += [self.tokenizer.token_to_id(word) for word in sentence]
                    label_ids += [self.label2id[label] for label in labels]
                    # [END]token
                    token_ids.append(self.tokenizer.token_to_id(self.tokenizer._token_end))
                    label_ids.append(self.label2id["O"])
                    segment_ids = [0 for _ in range(len(token_ids))]
                    data.append([[token_ids, segment_ids], [label_ids], len(sentence), text_index])

        df['label'] = df['label'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        for i, row in tqdm(df.iterrows(), desc='Prepare data'):
            text = row['text']
            label = row['label']
            label_seq = entity_tags_to_bio([label], [text])[0] # 序列标注，实体重叠默认保留靠前的实体
            sentence_lst = cut_sentence(text, max_len=self.config.max_len) # 获取切句结果
            # 转换切句标注
            labels_lst = []
            start = 0
            for sentence in sentence_lst:
                labels_lst.append(label_seq[start:start+len(sentence)])
                start += len(sentence)
            _add_data(sentence_lst, labels_lst, i)
        return data

    def load_model(self, _compile=False):
        try:
            self.model = load_model(self.config.model_path,
                                    custom_objects=self.custom_objects,
                                    compile=_compile)
        except:
            self.model = load_model(self.config.model_path,
                                    custom_objects=None,
                                    compile=_compile)
        self.model.summary()
        self.CRF = self.model.layers[-1]
        trans = K.eval(self.CRF.trans)
        # 这一步可视情况添加
        trans = self.suppress_impossible(trans)
        self.viterbi_decoder = ViterbiDecoder(trans=trans)

    def save_model(self, model_save=False, model=None):
        # 默认只在train过程中save模型
        if model_save:
            if model is None:
                model = self.model
            model.save(self.config.model_path, include_optimizer=False) # 模型中始终不存储optimizer

    def predict(self, contents: str):
        """预测一个句子"""
        return self.predict_batch([contents])[0]

    def batch_tokenize(self, text_list):
        sentences, tokens, segments, data_idx, start_ixs = [], [], [], [], []
        # 切句子，每句预测
        for idx, content in enumerate(text_list):
            cut_result = cut_sentence(content, max_len=self.config.max_len)
            s_ix = 0
            for text in cut_result:
                sentences.append(text)
                row_token = [self.tokenizer.token_to_id(self.tokenizer._token_start)]
                row_token += [self.tokenizer.token_to_id(word) for word in text]
                row_token.append(self.tokenizer.token_to_id(self.tokenizer._token_end))
                row_seg = [0 for _ in range(len(row_token))]
                tokens.append(row_token)
                segments.append(row_seg)
                data_idx.append(idx)
                start_ixs.append(s_ix)
                s_ix += len(text)
        tokens = sequence_padding(tokens)
        segments = sequence_padding(segments)
        return sentences, tokens, segments, data_idx, start_ixs

    def predict_batch(self, text_list):
        sentences, tokens, segments, data_idx, start_ixs = self.batch_tokenize(text_list)
        nodes_predict = self.model.predict([tokens, segments], batch_size=64)
        preds = [self.viterbi_decoder.decode(nodes) for nodes in nodes_predict]
        final_results = [[] for _ in range(len(text_list))]
        for row_id, row_result in enumerate(preds):
            sentence = sentences[row_id]
            temp_res_lst = tag_to_entity_list(sentence, [self.id2label[label_id] for label_id in row_result[1:]])
            for res_dic in temp_res_lst:
                res_dic["start"] += start_ixs[row_id]
                res_dic["end"] += start_ixs[row_id]
            final_results[data_idx[row_id]] += temp_res_lst
        return final_results

    def get_config(self):
        return self.config.get_para()


if __name__ == '__main__':
    df_data = pd.read_csv('cluener/train_per_loc_work.csv', dtype='object', encoding='utf-8')
    params = {
        "max_len": 128,
        "learning_rate": 2e-5,
        "epochs": 100,
        "batch_size": 32,
        "early_stopping_patience": 10,
        "val_size": None,
        "test_data_path": "cluener/dev_per_loc_work.csv",
        "model_path": "./model_cluener_crf",
    }
    model = LMModel(config=params, logger=None)
    model.train(df_data)
    model.save_model()
    config = model.get_config()
    print(config)
