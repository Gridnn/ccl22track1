### 1.简介
ngram+mlm(ngram参考kenlm实现)

### 2.目录代码介绍
- data
-- track1 赛道一测试数据
-- other 其他是字词词典数据
- model
-- chinese_L-12_H-768_A-12 bert_model用于mlm
-- ngram_model ngram训练保存的klm模型文件
- src
-- ngram_corrector.py ngram纠错
-- post_mlm.py mlm过滤
-- predict_ngram.py ngram预测代码
- track1_result.xlsx 结果文件

### 3.运行
#### 安装依赖
- pip3 install -r requirements.txt
#### 数据模型下载及解压
- https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
- unzip chinese_L-12_H-768_A-12.zip
- 微信语料
-- https://github.com/nonamestreet/weixin_public_corpus
- 比赛语料
-- https://github.com/blcuicall/CCL2022-CLTC/tree/main/datasets/track1
- 清洗生成train_ngram_char.txt(空格分隔-参考https://github.com/mattzheng/py-kenlm-model)
#### 训练
- docker pull 511023/kenlm:latest
- docker run -v data/train_ngram_char.txt:/root/ -d 511023/kenlm:latest
- bin/lmplz -o 4 --discount_fallback </root/train_ngram_char.txt> /root/text.arpa
- bin/build_binary -S 80% -s /root/text.arpa /root/text.klm
#### 预测
- python3 predict_ngram.py
