# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 12:03
# @Author  : Heng Li
# @File    : see_frozen_pb.py
# @Software: PyCharm

import tensorflow as tf
import os

frozen_pb = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\pb_dir\\ner_model.pb"

# 读取并创建一个图graph来存放Google训练好的Inception_v3模型（函数）
def create_graph():
    with tf.gfile.FastGFile(frozen_pb, 'rb') as f:
        # 使用tf.GraphDef()定义一个空的Graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')

# 创建graph
create_graph()

tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
result_file =  './result.txt'
with open(result_file, 'w+') as f:
    for tensor_name in tensor_name_list:
        f.write(tensor_name + '\n')


