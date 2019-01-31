import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import random


#  Read file
with open("/home/hinton/bhargav/active_qa/bert-as-service/ms_data/data1.tsv", 'r') as f:
    content = f.readlines()

ques_list = []
pos_paras_list = []
neg_paras_list = []
ans_list = []
print("Reading file..")
for i, x in enumerate(content):
    temp_lt = x.split("\t")
    try:
        neg_paras_list.append(temp_lt[2])
        pos_paras_list.append(temp_lt[1])
        ques_list.append(temp_lt[0])
    except:
        print(i)

print("Questions = {}".format(len(ques_list)))
print("Pos Paras = {}".format(len(pos_paras_list)))
print("Neg_paras = {}".format(len(neg_paras_list)))

# Reading USE module
print("Reading TF module")
module_url = "/home/hinton/bhargav/active_qa/bert-as-service/ms_data/use_module/96e8f1d3d4d90ce86b2db128249eb8143a91db73"
embd = hub.Module(module_url)

