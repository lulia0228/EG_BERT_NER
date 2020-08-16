#_*_ coding: utf-8 _*_
'''
@Author  : HengLi
@Time    : 18-12-17 下午5:12
@Software: PyCharm Community Edition

'''

'''-通过传入CKPT模型的路径得到模型的图和变量数据
-通过import_meta_graph导入模型中的图
-通过saver.restore从模型中恢复图中各个变量的数据
-通过graph_util.convert_variables_to_constants将模型持久化'''

import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from bert_base.server.helper import get_run_args
import numpy as np
import codecs
import pickle

def optimize_ner_model(args, num_labels):
    """
    加载中文NER模型
    :param args:
    :param num_labels:
    :param logger:
    :return:
    """
    # if not logger:
    #     logger = set_logger(colored('NER_MODEL, Lodding...', 'cyan'), args.verbose)
    # 如果PB文件已经存在则，返回PB文件的路径，否则将模型转化为PB文件，并且返回存储PB文件的路径
    if args.model_pb_dir is None:
        # 获取当前的运行路径
        tmp_file = os.path.join(os.getcwd(), 'predict_optimizer')
        if not os.path.exists(tmp_file):
            os.mkdir(tmp_file)
    else:
        tmp_file = args.model_pb_dir
    pb_file = os.path.join(tmp_file, 'ner_model.pb')
    if os.path.exists(pb_file):
        print('pb_file exits', pb_file)
        return pb_file

    import tensorflow as tf
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
            input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')

            from bert_base.bert import modeling
            bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'bert_config.json'))
            from bert_base.train.models import create_model
            (total_loss, logits, trans, pred_ids) = create_model(
                bert_config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, segment_ids=None,
                labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0, lstm_size=args.lstm_size)
            pred_ids = tf.identity(pred_ids, 'pred_ids')
            print("server.graph.py_line290: ",pred_ids.shape) # (?, 128)
            saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))
            from tensorflow.python.framework import graph_util
            # 这里是把输出节点给重新命名了：pred_ids = tf.identity(pred_ids, 'pred_ids') 即'pred_ids'
            tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_ids'])
            # from tensorflow.python.tools import freeze_graph

    # 存储二进制模型到文件中
    # logger.info('write graph to a tmp file: %s' % pb_file)
    with tf.gfile.GFile(pb_file, 'wb') as f:
        f.write(tmp_g.SerializeToString())
    return pb_file


# 不能以freeze_graph_test命名.会自动进入Nosetests环境
def freeze_graph_vtest(pb_path, sentence, tokenizer):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    def convert(line):
        feature = convert_single_example(0, line, label_list, args.max_seq_len, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids], (batch_size, args.max_seq_len))
        input_mask = np.reshape([feature.input_mask], (batch_size, args.max_seq_len))
        segment_ids = np.reshape([feature.segment_ids], (batch_size, args.max_seq_len))
        label_ids = np.reshape([feature.label_ids], (batch_size, args.max_seq_len))
        return input_ids, input_mask, segment_ids, label_ids

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 定义输入张量名
            input_tensor1 = sess.graph.get_tensor_by_name("input_ids:0")
            input_tensor2 = sess.graph.get_tensor_by_name("input_mask:0")
            # 定义输出张量
            output_tensor_name = sess.graph.get_tensor_by_name("pred_ids:0")
            # 处理输入
            sentence_token = tokenizer.tokenize(sentence)
            print('your input is:{}'.format(sentence_token))
            input_ids, input_mask, segment_ids, label_ids = convert(sentence_token)
            # print(input_ids)
            # exit()
            # run session get current feed_dict result
            out = sess.run(output_tensor_name, feed_dict={input_tensor1: input_ids, input_tensor2:input_mask })
            print("out:{}".format(out)) #     [2,5,9,11,7,12,9,9,9,1,9,9,9,9,9,9,9,3]

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
    args = get_run_args()
    label_list = ['B-MISC', '[CLS]', '[SEP]', 'I-LOC', 'B-LOC', 'I-MISC', 'I-PER', 'I-ORG', 'O', 'B-ORG', 'B-PER', 'X']
    num_labels = len(label_list) + 1
    print(args)
    model_dir = 'D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\output'
    bert_dir = 'D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\cased_L-12_H-768_A-12'

    with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}

    # rst = [2,5,9,11,7,12,9,9,9,1,9,9,9,9,9,9,9,3]
    # print([id2label[i] for i in rst ])
    # exit()

    from bert_base.bert import tokenization, modeling
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=False)

    # 生成frozen pb
    # optimize_ner_model(args, num_labels)
    batch_size = 1

    # 测试pb模型
    pb_file = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\pb_dir\\ner_model.pb"
    input_sentence = "Japan coach Shu Kamo said: The Syrian own goal proved lucky for us."
    freeze_graph_vtest(pb_file, input_sentence, tokenizer)

# ['[CLS]', 'B-LOC', 'O', 'B-PER', 'I-PER', 'X', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', '[SEP]']