# 实体纠错

## 文件说明
| 文件名 | 说明 |
| - | - |
| model_cluener_crf | 包含在cluener开源数据上事先训练好的Bert+CRF的序列标注模型及其配置组件 |
| entity_dict.json | 包含人名、地名、作品名的实体词典 |
| models.json | 序列标注模型使用的开源预训练模型配置信息 |
| ner_model.py | 序列标注模型类，包含训练和预测等方法 |
| ner.py | 实体纠错主流程代码 |
| whole_confusion.json | 中文字的音形近字混淆集词典 |

## 运行方法
### 实体纠错
```shell
python ner.py $src_file $trg_file $output_file
```
| 命令行参数 | 说明 |
| - | - |
| src_file | 测试数据文件路径，文件格式形如yaclc-csc-test.src|
| trg_file | 实体纠错前的纠错结果文件路径，文件格式为本次比赛测评的lbl格式 |
| output_file | 实体纠错后的纠错结果文件保存路径，文件格式为本次比赛测评的lbl格式 |
- ner.py 内部参数配置

| 参数名 | 默认值 | 说明 |
| - | - | - |
| ENTITY_DICT_PATH | "entity_dict.json" | 实体词典文件路径 |
| CONFUSION_DICT_PATH | "whole_confusion.json" | 音形字混淆集词典文件路径 |
| NER_MODEL_PATH | "model_cluener_crf" | 序列标注模型路径 |

### 序列标注模型训练
```shell
python ner_model.py
```
- ner_model.py 内部参数配置
```
params = {
    "max_len": 128, # 样本截断长度
    "learning_rate": 2e-5, # 学习率
    "epochs": 100, # 最大模型学习轮次数
    "batch_size": 32, # 批大小
    "early_stopping_patience": 10, # 早停机制等待轮数
    "test_data_path": "cluener/dev_per_loc_work.csv", # 验证数据文件路径
    "model_path": "./model_cluener_crf", # 模型保存路径
}
```