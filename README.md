# nlp-CGED
Chinese Grammatical Error Diagnosis</br>
中文语法纠错研究 基于序列标注的方法
# 所需环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.10.6</br>
笔者使用了开源的bert4keras，一个keras版的transformer模型库。bert4keras的更多介绍参见[这里](https://github.com/bojone/bert4keras)。
# 项目目录
├── bert4keras</br>
├── data    存放数据</br>
├── pretrained_model    存放预训练模型</br>
├── models    存放CRF等算法</br>
├── CGED_train.py    训练代码</br>
├── CGED_predict.py    评估和测试代码
# 数据集
数据集采用的CGED官方提供 转换为序列标注的形式，具体可以看data中的数据
# 使用说明
1.[下载预训练语言模型](https://github.com/google-research/bert#pre-trained-models)</br>
&emsp;&emsp;可采用BERT-Base, Chinese等模型</br>
&emsp;&emsp;更多的预训练语言模型可参见[bert4keras](https://github.com/bojone/bert4keras)给出的权重。</br>
2.构建数据集(数据集已处理好)</br>
&emsp;&emsp;train.json和test.json</br>
3.训练模型</br>
```
python CGED_train.py
```
4.评估和测试</br>
```
python CGED_predict.py
```
# 结果
| 数据集 | f1 | precision | recall |
| :------:| :------: | :------: | :------: |
| test | 0.46373 | 0.48993 | 0.44019 |


有任何问题欢迎私聊
