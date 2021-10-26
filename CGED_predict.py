#! -*- coding:utf-8 -*-
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0

import json
import numpy as np
from bert4keras.backend import K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from keras.models import Model
from tqdm import tqdm
import time

start = time.time()

# 基本信息
maxlen = 128
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率

# bert配置
config_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'pretrained_model/chinese_L-12_H-768_A-12/vocab.txt'

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            l = json.loads(line)
            if len(l['label']) == 0 or l['label'][0]['label'] == 'correct':
                D.append(
                    {
                        'text': l['raw_text'],
                        'labels': [{"label": "correct", "entity": ""}]
                    })
            else:
                D.append(
                    {
                        'text': l['raw_text'],
                        'labels': l['label']
                    }
                )
    return D

# 读取schema
with open('data/train.json', encoding='utf-8') as f:
    id2label, label2id= {}, {}
    for line in f:
        l = json.loads(line)
        for label in l['label']:
            if label['label'] not in label2id:
                id2label[len(label2id)] = label['label']
                label2id[label['label']] = len(label2id)

    num_labels = len(id2label) * 2 + 1
# print(label2id)
# print(id2label)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载预训练模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # model = "albert", #预训练模型选择albert时开启
)

from models.layers import CRFGraph

output, CRF = CRFGraph(model.output, num_labels, crf_lr_multiplier)
model = Model(model.input, output)
# model.summary()


def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[:, 0].argmax()]


def extract_arguments(text):
    """arguments抽取函数
    """
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 128:
        tokens.pop(-2)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    labels = viterbi_decode(nodes, trans)
    arguments, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                arguments.append([[i], id2label[(label - 1) // 2]])
            elif starting:
                arguments[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    return {
        text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
        for w, l in arguments
    }

def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    pbar = tqdm()
    for d in data:
        entity_pred = []
        entity_y = []
        pred_arguments = extract_arguments(d['text'])
        if len(pred_arguments) == 0:
            entity_pred.append(("correct", ""))
        for k, v in pred_arguments.items():
            entity_pred.append((v, k))
        for label_entity in d['labels']:
            entity_y.append((label_entity['label'], label_entity['entity']))
        R = set(entity_pred)
        T = set(entity_y)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))
    pbar.close()
    return f1, precision, recall

if __name__ == '__main__':
    # 加载模型
    model.load_weights('./save/best_model.weights')

    # 测试
    text1 = "当然吸烟者也会偶尔遇到他人的烟碰你的情况，不难想像，你也会讨厌的。"
    label2zh = {"R": "冗余", "M": "缺失", "S": "用词不当", "W": "无序词"}
    result = extract_arguments(text1)
    if len(result) == 0:
        print({"label": "correct"})
    else:
        result_list = [(label2zh[v], k) for k, v in result.items()]
        print(result_list)

    # 读取数据
    valid_data = load_data('data/test.json')
    print(len(valid_data))
    #
    # # 评估数据
    f1, precision, recall = evaluate(valid_data)
    print('f1: %.5f, precision: %.5f, recall: %.5f\n' % (f1, precision, recall))


    delta_time = time.time()-start
    print("时间：", delta_time)