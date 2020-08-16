# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 13:37
# @Author  : Heng Li
# @File    : export_saved_pb.py
# @Software: PyCharm

'''
BERT模型ckpt文件转为部署tensorflow-serving所需文件
'''
import json
import os
from enum import Enum
from termcolor import colored
import sys
import logging
import tensorflow as tf
import argparse
import pickle
import shutil

tf.app.flags.DEFINE_string('export_model_dir', "output_1", 'Directory where the model exported files should be placed.')
tf.app.flags.DEFINE_integer('model_version', 1, 'Models version number.')
FLAGS = tf.app.flags.FLAGS

def main(max_seq_len, model_dir, num_labels):

    with tf.Session() as sess:
        #输入占位符
        input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')
        #模型前向传播
        from bert_base.bert import modeling
        bert_config_file = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\cased_L-12_H-768_A-12\\bert_config.json"

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        from bert_base.train.models import create_model

        (total_loss, logits, trans, pred_ids) = create_model(
            bert_config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, segment_ids=None,
            labels=None, num_labels=num_labels, use_one_hot_embeddings=False)
        pred_ids = tf.identity(pred_ids, 'pred_ids')
        print("server.graph.py_line290: ", pred_ids.shape)  # (?, 128)
        saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess,latest_checkpoint )
        # Create SavedModelBuilder class
        # defines where the model will be exported
        export_path_base = FLAGS.export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        if os.path.exists(export_path):
            shutil.rmtree(export_path)

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
        input_ids_tensor = tf.saved_model.utils.build_tensor_info(input_ids)
        input_mask_tensor = tf.saved_model.utils.build_tensor_info(input_mask)
        # output tensor info
        pred_ids_output = tf.saved_model.utils.build_tensor_info(pred_ids)

        # Defines the DeepLab signatures, uses the TF Predict API
        # It receives an image and its dimensions and output the segmentation mask
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input_ids': input_ids_tensor, 'input_mask': input_mask_tensor},
                outputs={'pred_label': pred_ids_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'result':
                    prediction_signature,
            })
        # export the model
        # builder.save(as_text=True) # saved_model.pbtxt
        builder.save()  # saved_model.pb
        print('Done exporting!')

if __name__ == '__main__':
    max_seq_len = 128
    num_labels = 13
    model_dir = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\output"
    main(max_seq_len, model_dir, num_labels)