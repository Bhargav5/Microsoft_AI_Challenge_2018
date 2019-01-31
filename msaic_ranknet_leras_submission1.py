import numpy as np
from keras.layers import Input, Dense, Dropout, Subtract, Activation
from keras.models import Model

with open(u'/home/hinton/bhargav/active_qa/bert-as-service/ms_data/eval2_unlabelled.tsv') as f:
    content = f.readlines()

id_list=[]

print("Reading File")
for i,x in enumerate(content):
    temp_lt = x.split("\t")
    id_list.append(temp_lt[0])

id_list1 = [id_list[i] for i in range(0,len(id_list),10)]
ques_vec = np.load("eval2_unlabelled.tsv.qn.npy")
paras_vec = np.load("eval2_unlabelled.tsv.psg.npy")

ques_vec = np.reshape(ques_vec, newshape=(-1, 10, 512))
paras_vec = np.reshape(paras_vec, newshape=(-1,10,512))
print("File has been read")
print("Id list len={}\nQuestion Len={}\nParas Len={}".format(len(id_list1),ques_vec.shape,paras_vec.shape))

def basic_model(input_dims=512):

    inputs = Input(shape=(input_dims,), dtype="float32")
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(rate=0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    y = Dense(1)(x)
    model = Model(inputs=inputs, outputs=y)

    return model


base_model = basic_model(input_dims=512)

inp1 = Input(shape=(512,), dtype='float32', name='Correct_Paragraph')  # Always some variation of True paragraph and Question
inp2 = Input(shape=(512,), dtype='float32', name='Original_Paragraph')  # Some variation of False + True paragraph and Question

y1 = base_model(inp1)
y2 = base_model(inp2)
y = Subtract()([y1, y2])
y_out = Activation("sigmoid")(y)

final_model = Model([inp1, inp2], y_out)
final_model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Load weights
print("Loading weights...")
final_model.load_weights("absolute_diff_model_weight_0_drop.h5")

base_model1 = basic_model(input_dims=512)
inp11 = Input(shape=(512,), dtype='float32', name='Correct_Paragraph')  # Always some variation of True paragraph and Question
inp21 = Input(shape=(512,), dtype='float32', name='Original_Paragraph')  # Some variation of False + True paragraph and Question

y11 = base_model1(inp11)
y21 = base_model1(inp21)
y1 = Subtract()([y11, y21])
y_out1 = Activation("sigmoid")(y1)

final_model1 = Model([inp11, inp21], y_out1)
final_model1.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Load weights
print("Loading weights...")
final_model1.load_weights("absolute_diff_model_weight.h5")

import csv
tsvfile=open('answer.tsv','w')
writer=csv.writer(tsvfile, delimiter='\t')

for i in range(len(id_list1)):
    print(i)
    temp_id = id_list1[i]
    q = ques_vec[i]
    p = paras_vec[i]
    q = np.reshape(q, newshape=(-1, 512))
    p = np.reshape(p, newshape=(-1, 512))
    diff = q - p
    pred1 = base_model.predict(diff)
    pred2 = base_model1.predict(diff)
    preds = pred2 + pred2
    out_lt = preds.reshape(-1,).tolist()
    ip_lt = [temp_id] + out_lt
    writer.writerow(ip_lt)
tsvfile.close()