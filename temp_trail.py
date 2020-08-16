# -*- coding: utf-8 -*-
# @Time    : 2020/5/26 19:02
# @Author  : Heng Li
# @File    : temp_trail.py
# @Software: PyCharm
import  pickle
import codecs
import os

'''file_path = 'D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\NERdata'

with open(os.path.join(file_path, "dev.txt"), 'r') as rf:
    tmp = rf.readlines()

with open("./dev.txt", 'w') as file:
    for i in tmp:
        if i == '\n':
            file.write(i)
        else:
            file.write(i.split(" ")[0]+" "+i.split(" ")[-1])'''


with codecs.open(r'D:\Program Files\JetBrains\PyCharm 2017.2.4\bert_ner_3\output\label_list.pkl', 'rb') as rf:
# with codecs.open(r'D:\Program Files\JetBrains\PyCharm 2017.2.4\bert_ner_3\output\label2id.pkl', 'rb') as rf:
    labels = pickle.load(rf)

print(labels)

# {'B-MISC', 'I-ORG', 'B-ORG', 'O', 'I-LOC', 'B-PER', 'X', 'B-LOC', 'I-PER', 'I-MISC', '[CLS]', '[SEP]'}
# {'B-MISC': 1, '[CLS]': 2, '[SEP]': 3, 'I-LOC': 4, 'B-LOC': 5, 'I-MISC': 6, 'I-PER': 7, 'I-ORG': 8, 'O': 9, 'B-ORG': 10, 'B-PER': 11, 'X': 12}
label_list = ['B-MISC',  '[CLS]',  '[SEP]',  'I-LOC',  'B-LOC',  'I-MISC', 'I-PER',  'I-ORG',  'O', 'B-ORG', 'B-PER', 'X']
label_map = {}
for (i, label) in enumerate(label_list, 1):
    label_map[label] = i
print(label_map)
