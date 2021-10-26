#! -*- coding:utf-8 -*-
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.models import Model
from tqdm import tqdm

# 基本信息
maxlen = 128
epochs = 10
batch_size = 32
learning_rate = 1e-5
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

# 读取数据
train_data = load_data('data/train.json')
print(len(train_data))

valid_data = load_data('data/test.json')
print(len(valid_data))

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


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)
            labels = [0] * len(token_ids)
            for labels_entity in d['labels']:
                if labels_entity['label'] != 'correct':
                    a_token_ids = tokenizer.encode(labels_entity['entity'])[0][1:-1]
                    start_index = search(a_token_ids, token_ids)
                    if start_index != -1:
                        labels[start_index] = label2id[labels_entity['label']] * 2 + 1
                        for i in range(1, len(a_token_ids)):
                            labels[start_index + i] = label2id[labels_entity['label']] * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

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

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)


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


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists('save'):
            os.makedirs('save')
        if epoch >= 0:
            f1, precision, recall = evaluate(valid_data)
            if f1 >= self.best_val_f1:
                self.best_val_f1 = f1
                model.save_weights('./save/best_model.weights')
            print(
                'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
                (f1, precision, recall, self.best_val_f1)
            )


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )