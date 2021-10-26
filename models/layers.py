import keras
from bert4keras.backend import K
from keras.layers import LSTM, Dropout, Bidirectional, Dense, GRU, Conv1D, Concatenate
from bert4keras.layers import ConditionalRandomField

### SelfAttention #######################
class SelfAttention(keras.layers.Layer):
    """
        self attention,
        codes from:  https://mp.weixin.qq.com/s/qmJnyFMkXVjYBwoR_AQLVA
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        # W、K and V
        self.kernel = self.add_weight(name='WKV',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      regularizer=keras.regularizers.l1_l2(0.0000032),
                                      trainable=True)
        super().build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        # print("WQ.shape",WQ.shape)
        # print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64**0.5)
        QK = K.softmax(QK)
        # print("QK.shape",QK.shape)
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = {"output_dim": self.output_dim,}
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def CRFGraph(model_output, num_labels, crf_lr_multiplier):
    # CRF(crf of bert4keras, 条件概率随机场)
    # Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data(https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)

    output = Dense(num_labels)(model_output)
    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)
    return output, CRF

def BiLstmCRFGraph(model_output, num_labels, crf_lr_multiplier):
    # Bi - LSTM - CRF
    #Bidirectional LSTM-CRF Models for Sequence Tagging(https://arxiv.org/pdf/1508.01991.pdf)

    #rnn_unit:RNN隐藏层, 8的倍数, 一般取64, 128, 256, 512, 768等
    #activate_mid:中间激活函数, 非线性变幻, 提升逼近能力, 选择"relu","tanh"或"sigmoid"
    #activate_end:结束激活函数, 即最后一层的激活函数

    rnn_unit = 256
    num_rnn_layers = 1 #1,2,3
    dropout = 0.5
    use_crf = True
    activate_mid = 'tanh'
    activate_end = 'softmax'
    rnn_cell = LSTM #GRU, LSTM
    for nrl in range(num_rnn_layers):
        x = Bidirectional(rnn_cell(units=rnn_unit,
                               return_sequences=True,
                               activation=activate_mid,
                               ))(model_output)
        # x = Dropout(dropout)(x)
    if use_crf:
        x = Dense(units=num_labels)(x)#, activation=activate_end
        CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        output = CRF(x)
    return output, CRF

def CnnLstmGraph(model_output, num_labels, crf_lr_multiplier):
    # Bi - LSTM - CNNs - CRF, 有改动, 为更适普的"卷积神经网络+循环神经网络", CNN + LSTM
    #Bidirectional LSTM-CRF Models for Sequence Tagging(https://arxiv.org/pdf/1508.01991.pdf)

    rnn_cell = GRU  # GRU,LSTM
    filters_size = [3, 4, 5]  # 卷积核尺寸, 1-10
    activate_mid = 'tanh'
    activate_end = 'softmax'
    filters_num = 300  # 卷积个数 text-cnn:300-600
    rnn_unit = 256
    dropout = 0.5
    use_crf = True

    # CNN-LSTM, 提取n-gram特征和最大池化， 一般不用平均池化
    conv_pools = []
    for i in range(len(filters_size)):
        conv = Conv1D(name="conv-{0}-{1}".format(i, filters_size[i]),
                      kernel_size=filters_size[i],
                      activation=activate_mid,
                      filters=filters_num,
                      padding='same',
                      )(model_output)
        conv_rnn = Bidirectional(rnn_cell(name="bi-lstm-{0}-{1}".format(i, filters_size[i]),
                                          activation=activate_mid,
                                          return_sequences=True,
                                          units=rnn_unit, )
                                 )(conv)
        x_dropout = Dropout(rate=dropout, name="dropout-{0}-{1}".format(i, filters_size[i]))(conv_rnn)
        conv_pools.append(x_dropout)
    # 拼接
    x = Concatenate(axis=-1)(conv_pools)
    x = Dropout(dropout)(x)
    # CRF or Dense
    if use_crf:
        x = Dense(units=num_labels)(x)#, activation=activate_end
        CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        output = CRF(x)
    return output, CRF

def BiLstmLANGraph(model_output, num_labels, crf_lr_multiplier):
    # Bi - LSTM - LAN(双向 - 长短时记忆神经网络 + 注意力机制)
    # Hierarchically-Reﬁned Label Attention Network for Sequence Labeling(https://arxiv.org/abs/1908.08676v2)

    rnn_cell = LSTM  #GRU,LSTM
    num_rnn_layers = 1 #1,2,3
    rnn_unit = 256
    activate_mid = 'tanh'
    activate_end = 'softmax'
    dropout = 0.5
    use_crf = True

    # Bi-LSTM-LAN
    for nrl in range(num_rnn_layers):
        x = Bidirectional(rnn_cell(units=rnn_unit*(nrl+1),
                                     return_sequences=True,
                                     activation=activate_mid,
                                     ))(model_output)
        x_att = SelfAttention(K.int_shape(x)[-1])(x)
        outputs = Concatenate()([x, x_att])
        # outputs = Dropout(dropout)(outputs)
    if use_crf:
        x = Dense(units=num_labels)(outputs)#, activation=activate_end
        CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        output = CRF(x)
    return output, CRF

def DGCNNGraph(model_output, num_labels, crf_lr_multiplier):
    # DGCNN(Dilate Gated Convolutional Neural Network, 即"膨胀门卷积神经网络", IDCNN + CRF)
    # Multi-Scale Context Aggregation by Dilated Convolutions(https://arxiv.org/abs/1511.07122)
    # CNN, 提取n-gram特征和最大池化, DGCNN膨胀卷积(IDCNN)

    filters_size = [3, 4, 5]  # 卷积核尺寸, 1-10
    atrous_rates = [2, 1, 2]
    activate_mid = 'tanh'
    activate_end = 'softmax'
    filters_num = 300  # 卷积个数 text-cnn:300-600
    dropout = 0.5
    use_crf = True
    conv_pools = []
    for i in range(len(filters_size)):
        conv = Conv1D(name="conv-{0}-{1}".format(i, filters_size[i]),
                        dilation_rate=atrous_rates[0],
                        kernel_size=filters_size[i],
                        activation=activate_mid,
                        filters=filters_num,
                        padding="SAME",
                        )(model_output)
        for j in range(len(atrous_rates) - 1):
            conv = Conv1D(name="conv-{0}-{1}-{2}".format(i, filters_size[i], j),
                            dilation_rate=atrous_rates[j],
                            kernel_size=filters_size[i],
                            activation=activate_mid,
                            filters=filters_num,
                            padding="SAME",
                            )(conv)
            conv = Dropout(name="dropout-{0}-{1}-{2}".format(i, filters_size[i], j),
                             rate=dropout,)(conv)
        conv_pools.append(conv)
    # 拼接
    x = Concatenate(axis=-1)(conv_pools)
    x = Dropout(dropout)(x)
    # CRF or Dense
    if use_crf:
        x = Dense(units=num_labels)(x)#, activation=activate_end
        CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        output = CRF(x)
    return output, CRF