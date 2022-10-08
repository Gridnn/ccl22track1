# 困惑度减少误召回

该模块通过计算句子的困惑度，对比修改前和修改后的困惑度来减少模型的误召回。

## 文件说明
- fluent.py 困惑度减少误召回主流程代码
- model.py 困惑度模型文件

使用方法
```angular2html
python3 fluent.py
```
其中参数threshold可以在fluent.py内部进行调节。