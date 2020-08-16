# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 0:05
# @Author  : Heng Li
# @File    : my_client.py
# @Software: PyCharm

from __future__ import print_function

# docker run -itd -p 9000:8500 -p 9001:8501 tf_bert_ner_serving:0601-1

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from keras.preprocessing import image as I

import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
import os

tf.app.flags.DEFINE_string('server', '127.0.0.1:9000', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

def main(sentence, tokenizer):
    # host, port = FLAGS.server.split(':')
    # channel = implementations.insecure_channel(host, int(port))
    # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    #  Send request

    def convert(line):
        feature = convert_single_example(0, line, label_list, 128, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids], (batch_size,128))
        input_mask = np.reshape([feature.input_mask], (batch_size, 128))
        segment_ids = np.reshape([feature.segment_ids], (batch_size, 128))
        label_ids = np.reshape([feature.label_ids], (batch_size, 128))
        return input_ids, input_mask, segment_ids, label_ids

    # 处理输入
    sentence_token = tokenizer.tokenize(sentence)
    print('your input is:{}'.format(sentence_token))
    input_ids, input_mask, segment_ids, label_ids = convert(sentence_token)

    # input_ids = np.expand_dims(input_ids, axis=0)
    # input_mask = np.expand_dims(input_mask, axis=0)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'bert_ner' # 这个name跟tensorflow_model_server --model_name="bert_ner" 对应
    request.model_spec.signature_name = 'result' # 这个signature_name 跟signature_def_map 对应

    request.inputs['input_ids'].CopyFrom(
         tf.contrib.util.make_tensor_proto(input_ids, shape=[input_ids.shape[0], input_ids.shape[1]])) # shape跟 keras的model.input类型对应

    request.inputs['input_mask'].CopyFrom(
         tf.contrib.util.make_tensor_proto(input_mask, shape=[input_mask.shape[0], input_mask.shape[1]]))

    result_future = stub.Predict(request, 10.0) # 10 secs timeout
    # print("result_future",result_future)

    response1 = np.array(result_future.outputs['pred_label'].int_val)
    print("label_outcome: ", response1)
    print(len(response1))

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)

    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    from bert_base.train.models import InputFeatures
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature

if __name__ == '__main__':
    model_dir = 'D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\output'
    bert_dir = 'D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\cased_L-12_H-768_A-12'
    label_list = ['B-MISC', '[CLS]', '[SEP]', 'I-LOC', 'B-LOC', 'I-MISC', 'I-PER', 'I-ORG', 'O', 'B-ORG', 'B-PER', 'X']
    input_sentence = "Japan coach Shu Kamo said: The Syrian own goal proved lucky for us."
    # input_sentence = "RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR."
    batch_size = 1
    from bert_base.bert import tokenization, modeling
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=False)
    main(input_sentence, tokenizer)