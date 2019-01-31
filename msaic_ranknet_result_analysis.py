import numpy as np
from keras.layers import Input, Dense, Dropout, Subtract, Activation
from keras.models import Model

with open("/home/hinton/bhargav/active_qa/bert-as-service/ms_data/data_pairwise_1.tsv", 'r') as f:
    content = f.readlines()
ans_list = []

print("Reading file content..")
for i, x in enumerate(content):
    temp_lt = x.split("\t")
    ans_list.append(float(temp_lt[2]))
answers1_list = [ans_list[i:i + 10] for i in range(0, len(ans_list), 10)]
del (content)
del (ans_list)
right_para_index = []  # index of correct paragraph

for x in answers1_list:
    inx = x.index(1.0)
    right_para_index.append(inx)

print("Reading Questions")
questions = np.load("Questions_use.npy")
#questions = np.random.rand(10000, 10, 512)
print("Reading Paragraphs")
paragraphs = np.load("Paragraphs_use.npy")

print("File has been read")
print("Id list len={}\nQuestion Len={}\nParas Len={}".format(len(answers1_list),questions.shape,paragraphs.shape))

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
final_model.load_weights("absolute_diff_model_weight.h5")

right_answer_rank = []

for i in range(len(answers1_list)):
    print(i)
    q = questions[i]
    p = paragraphs[i]
    q = np.reshape(q, newshape=(-1, 512))
    p = np.reshape(p, newshape=(-1, 512))
    diff = q - p
    preds = base_model.predict(diff)
    #preds = preds / np.max(preds)
    #dist = np.linalg.norm(q-p, axis=1, keepdims=True)
    #dist = dist / np.max(dist)
    pred1 = np.argsort(preds.reshape(-1,))
    rank = 10 - (pred1.tolist().index(right_para_index[i]))
    right_answer_rank.append(rank)

arr = np.array(right_answer_rank, dtype=np.int32)
np.save("training_rank_analysis_drop_30.npy", arr)