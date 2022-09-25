### 简介
ngram+mlm

### 目录代码介绍
- data
-- track1 赛道一测试数据
-- txt 来源比赛及微信数据
- model
-- chinese_L-12_H-768_A-12 bert_model用于mlm
-- leveldb ngram保存的db模型文件
- src
-- ngram_corrector.py ngram纠错
-- post_mlm.py mlm过滤
-- predict_ngram.py ngram预测代码
-- train_ngram.py ngram无监督训练代码
- static
-- word_simi 字形混淆集
- track1_result.xlsx 结果文件

#### 运行
- 安装依赖
pip3 install -r requirements.txt
- 数据模型下载及解压
https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip
微信语料
https://github.com/nonamestreet/weixin_public_corpus
比赛语料
https://github.com/blcuicall/CCL2022-CLTC/tree/main/datasets/track1

- 训练
python3 train_ngram.py
- 预测
python3 predict_ngram.py
