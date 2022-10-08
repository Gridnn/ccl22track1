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

2. pipeline.sh中将会调用data_preprocess.py和train_pipeline函数，以进行对基础模型的训练。
```
sh pipeline.sh
```

### 推理
```angular2html
sh decode.sh
```