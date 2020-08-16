# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 23:52
# @Author  : Heng Li
# @File    : bert_ber_client.py
# @Software: PyCharm

# docker run -itd -p 9000:8500 -p 9001:8501 tf_bert_ner_serving:0601-1

import sys
sys.path.append('..')
import requests
import json
import os
import numpy as np

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

def bert_ner(sentence,tokenizer ):
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

    input_ids_list = input_ids.tolist()
    input_mask_list = input_mask.tolist()

    url = 'http://127.0.0.1:9001/v1/models/bert_ner:predict'
    data = json.dumps(
            {
                    "name": 'bert_ner',
                    "signature_name":'result',
                    "inputs":{
                            'input_ids': input_ids_list,
                            'input_mask': input_mask_list}})
    # result = requests.post(url, data=data).json()
    result = requests.post(url, data=data)
    res = json.loads(result.text)
    print(res)
    return result


if __name__ == '__main__':
    model_dir = 'D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\output'
    bert_dir = 'D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\cased_L-12_H-768_A-12'
    label_list = ['B-MISC', '[CLS]', '[SEP]', 'I-LOC', 'B-LOC', 'I-MISC', 'I-PER', 'I-ORG', 'O', 'B-ORG', 'B-PER', 'X']
    input_sentence = "Japan coach Shu Kamo said: The Syrian own goal proved lucky for us."
    batch_size = 1
    from bert_base.bert import tokenization, modeling
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=False)
    bert_ner(input_sentence, tokenizer)


