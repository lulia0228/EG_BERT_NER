# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 13:58
# @Author  : Heng Li
# @File    : example.py
# @Software: PyCharm

import requests
import os
import json

PREDICT_URL = "http://127.0.0.1:12345/ner_predict_service"
# PREDICT_URL = "http://192.168.197.1:12345/ner_predict_service"

if __name__ == '__main__':
    trail_str = "Japan coach Shu Kamo said: The Syrian own goal proved lucky for us."
    trail_dict ={}
    trail_dict['query'] = trail_str
    r = requests.post(PREDICT_URL, data=trail_dict)
    res_str = r.text
    res_dict = json.loads(r.text)
    print(res_dict['data'])


# [['B-LOC', 'O', 'B-PER', 'I-PER', 'X', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]  # simple_flask_http_service
# [['B-LOC', 'O', 'B-PER', 'I-PER', 'X', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]  # terminal_predict
# ['[CLS]', 'B-LOC', 'O', 'B-PER', 'I-PER', 'X', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', '[SEP]'] # freeze_inference_pb