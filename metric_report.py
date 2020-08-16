# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 14:35
# @Author  : Heng Li
# @File    : run_conlleval.py
# @Software: PyCharm

from sklearn.metrics import classification_report
res_file = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_3\\output\\label_test.txt"
# res_file = "D:\\Program Files\\JetBrains\\PyCharm 2017.2.4\\bert_ner_2\\output_bak\\result_dir\\label_test.txt"
label_dict = { 'O': 0, 'B-MISC': 1, 'I-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8}
label_list = [ 'O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
y_true = []
y_pred = []

with open(res_file, 'r',encoding="utf-8") as f:
    tmp_li = f.readlines()
    tmp_li = [i.strip() for i in tmp_li]
    for i in tmp_li :
        c = i.split("\t")
        if(len(c) < 3):
            print(c)
        else:
            if c[2] != '[SEP]':
                y_true.append(label_dict[c[1]])
                y_pred.append(label_dict[c[2]])
            else :
                print(c[2])

print(classification_report(y_true, y_pred, target_names=label_list))


'''y_true = []
y_pred = []
label_dict = { 'O': 0, 'B-MISC': 1, 'I-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8}
with open(res_file, 'r',encoding="utf-8") as f:
    tmp_li = f.readlines()
    tmp_li = [i.strip() for i in tmp_li]
    for i in range(len(tmp_li)) :
        c = tmp_li[i].split("\t")
        if(c[1] == 'O'):
            if c[2] != '[SEP]':
                y_true.append([label_dict[c[1]]])
                y_pred.append([label_dict[c[2]]])
            else:
                print(c[2])
        else:
            if(c[1].startswith('B')):
                y_true.append([label_dict[c[1]]])
                y_pred.append([label_dict[c[2]]])
            else:
                y_true[-1].append(label_dict[c[1]])
                y_pred[-1].append(label_dict[c[2]])

y_concat_true = []
y_concat_pred = []
concat_dict = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4}
concat_list = ['O', 'MISC', 'PER', 'ORG', 'LOC']
for i in range(len(y_pred)):
    # print("   ",y_true[i], " ", y_pred[i])
    if y_pred[i] == y_true[i]:
        y_concat_true.append(concat_dict[y_true[i][0]])
        y_concat_pred.append(concat_dict[y_true[i][0]])
    else:
        if len(y_true[i]) == 1 :
            y_concat_true.append(concat_dict[y_true[i][0]])
            y_concat_pred.append(concat_dict[y_pred[i][0]])
        else:
            y_concat_true.append(concat_dict[y_true[i][0]])
            y_concat_pred.append(0)

print(classification_report(y_concat_true, y_concat_pred, target_names=concat_list))

# ner2
#               precision    recall  f1-score   support
#
#            O       0.99      0.99      0.99     34069
#       B-MISC       0.80      0.66      0.73       668
#       I-MISC       0.63      0.56      0.59       194
#        B-PER       0.94      0.94      0.94      1311
#        I-PER       0.97      0.97      0.97       896
#        B-ORG       0.80      0.82      0.81      1191
#        I-ORG       0.77      0.80      0.78       657
#        B-LOC       0.89      0.88      0.89      1387
#        I-LOC       0.82      0.70      0.75       236
#
#     accuracy                           0.97     40609
#    macro avg       0.84      0.81      0.83     40609
# weighted avg       0.97      0.97      0.97     40609
#               precision    recall  f1-score   support

# ner3
#               precision    recall  f1-score   support
#
#            O       0.99      0.99      0.99     34046
#       B-MISC       0.86      0.68      0.76       668
#       I-MISC       0.66      0.60      0.63       194
#        B-PER       0.96      0.93      0.95      1308
#        I-PER       0.95      0.97      0.96       896
#        B-ORG       0.82      0.79      0.81      1189
#        I-ORG       0.79      0.81      0.80       657
#        B-LOC       0.85      0.91      0.87      1385
#        I-LOC       0.78      0.74      0.76       236

#     accuracy                           0.97     40579
#    macro avg       0.85      0.82      0.84     40579
# weighted avg       0.97      0.97      0.97     40579


# ner2
#            O       0.98      0.99      0.99     34070
#         MISC       0.71      0.69      0.70       668
#          PER       0.95      0.92      0.94      1311
#          ORG       0.84      0.71      0.77      1191
#          LOC       0.88      0.88      0.88      1387
#
#     accuracy                           0.97     38627
#    macro avg       0.87      0.84      0.85     38627
# weighted avg       0.97      0.97      0.97     38627

# ner3
#            O       0.99      0.99      0.99     34046
#         MISC       0.84      0.67      0.75       668
#          PER       0.96      0.93      0.95      1308
#          ORG       0.80      0.76      0.78      1189
#          LOC       0.88      0.89      0.88      1385
#
#     accuracy                           0.97     38596
#    macro avg       0.89      0.85      0.87     38596
# weighted avg       0.97      0.97      0.97     38596'''
