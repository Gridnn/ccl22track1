# 基于拼音编码和多轮纠错式推理方式的中文拼写检查纠错系统
## 目录代码介绍
- csc 基础模型
- data 训练用的数据
- data_generator 自动数据生成
- fluent 困惑度减少无纠错
- ner ner纠错方法
- ngram ngram纠错方法
- pymodel 拼音编码基础模型
- data_preprocess.py 数据预处理
- dataset.py 数据处理和储存
- decode.py 模型推理
- decode.sh 模型推理流水线
- eval_char_level 字符级别结果指标计算
- eval_sent_level 句子级别结果指标计算
- pipeline.sh 训练拼音编码基础模型流水线
- requirements.txt 环境要求
- train_pipeline.py 训练拼音编码基础模型
- utils.py 一些一般常用代码

## 1. 准备工作
### 下载环境
```angular2html
pip3 install -r requirements.txt
```

### 下载数据和模型
按文档附带的链接下载数据和模型，解压并保存至相关文件夹。

## 2. 使用模型

### 训练
1. 使用data_generator将维基百科语料，微信语料，新闻语料进行纠错数据对构造。

2. pipeline.sh中将会调用data_preprocess.py和train_pipeline.py函数，以进行对基础模型的训练。
```
sh pipeline.sh
```

#### 参数解释

##### data_preprocess.py
- source_dir 含错别字的数据地址
- target_dir 纠错后的数据地址
- bert_path 预训练模型地址
- save_path 预处理后数据输出地址
- data_mode 数据形式，para或lbl
- normalize 是否正则化

##### train_pipeline.py
- pretrained_model 预训练模型地址
- train_path 训练集数据地址
- dev_path 验证集数据地址
- test_path 测试集数据地址
- lbl_path 验证集答案
- test_lbl_path 测试集答案
- save_path 模型保存地址
- batch_size 批大小
- num_epochs 最大模型学习轮次数
- lr 学习率
- tag 标签



### 推理
```angular2html
sh decode.sh
```