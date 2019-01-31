'''
Training wiht top k semantic similar paragraphs
'''

import numpy as np
from keras.layers import Input, Dense, Dropout, Subtract, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint

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

# invert the answers
inverted_answers = []
for x in answers1_list:
    new_lt = [1.0 - z for z in x]
    inverted_answers.append(new_lt)

print("Inverted Ans Len = {}, Right_index_Len = {}".format(len(inverted_answers), len(right_para_index)))

print("Reading Questions")
questions = np.load("Questions_use.npy")
# questions = np.random.rand(10000, 10, 512)
print("Reading Paragraphs")
paragraphs = np.load("Paragraphs_use.npy")
# paragraphs = np.random.rand(10000, 10, 512)
print("Reading Answer Indexes..")
answer_indexes = np.load("training_ranknet_first_filter_topk.npy")

# inverted_answers = inverted_answers[:10000]
# right_para_index = right_para_index[:10000]

assert len(inverted_answers) == len(right_para_index) == len(questions) == len(paragraphs)
initial = 0
training_queries = 450000
# training_queries = 9500
batch_size = 512
top = 3

train_ques = questions[: training_queries]
train_paras = paragraphs[: training_queries]
train_ans = inverted_answers[: training_queries]
train_right_inx = right_para_index[: training_queries]
train_answer_indexes = answer_indexes[: training_queries]

test_ques = questions[training_queries:]
test_paras = paragraphs[training_queries:]
test_ans = inverted_answers[training_queries:]
test_right_inx = right_para_index[training_queries:]
test_answer_indexes = answer_indexes[training_queries:]

print("Train Questions = {} Train Paragraphs = {} Train Answers = {}".format(train_ques.shape, train_paras.shape,
                                                                             len(train_ans)))
print("Test Questions = {} Test Paragraphs = {} Test Answers = {}".format(test_ques.shape, test_paras.shape,
                                                                          len(test_ans)))


def data_generator(ques, paras, ans, ans_inx, answer_indexes, topk=top, batch=batch_size):
    if len(ques) % batch == 0:
        steps = len(ques) // batch
    else:
        steps = (len(ques) // batch) + 1
    while 1:
        for i in range(steps):
            i1 = i * batch
            # Read New data
            if i == steps - 1:
                q = ques[i1: len(ques)]
                p = paras[i1: len(ques)]
                a = ans[i1: len(ques)]
                inx = ans_inx[i1: len(ques)]
                ans_indexes = answer_indexes[i1: len(ques)]
            else:
                q = ques[i1: i1 + batch]
                p = paras[i1: i1 + batch]
                a = ans[i1: i1 + batch]
                inx = ans_inx[i1: i1 + batch]
                ans_indexes = answer_indexes[i1: i1 + batch]
            a = np.array(a, dtype=np.float32)
            # Create new array of correct paragraphs
            for j in range(len(p)):
                if j == 0:
                    right_para = np.tile(p[j][inx[j]], topk)
                    right_para = np.reshape(right_para, newshape=(-1, 512))
                else:
                    temp = np.tile(p[j][inx[j]], topk)
                    temp = np.reshape(temp, newshape=(-1, 512))
                    right_para = np.vstack((right_para, temp))
            # Create new array of only top 4 candidates
            for j in range(len(p)):
                for k in range(topk):
                    if j == 0 and k == 0:
                        q1 = q[j][ans_indexes[j][-(k+1)]]
                        p1 = p[j][ans_indexes[j][-(k+1)]]
                        a1 = a[j][ans_indexes[j][-(k+1)]]
                    else:
                        q1 = np.vstack((q1, q[j][ans_indexes[j][-(k+1)]]))
                        p1 = np.vstack((p1, p[j][ans_indexes[j][-(k+1)]]))
                        a1 = np.vstack((a1, a[j][ans_indexes[j][-(k+1)]]))


            q = np.reshape(q1, newshape=(-1, 512))
            p = np.reshape(p1, newshape=(-1, 512))
            a = np.reshape(a1, newshape=(-1, 1))
            '''
            para_diff = np.square((q - p))
            right_para_diff = np.square((q - right_para))

            para_diff = np.multiply(q, p)
            right_para_diff = np.multiply(q, right_para)
            '''
            right_para_diff = q - right_para
            para_diff = q - p

            yield [right_para_diff, para_diff], a


training_generator = data_generator(train_ques, train_paras, train_ans, train_right_inx, answer_indexes=train_answer_indexes)
validation_generator = data_generator(test_ques, test_paras, test_ans, test_right_inx, answer_indexes=test_answer_indexes)

if training_queries % batch_size == 0:
    training_steps = training_queries // batch_size
else:
    training_steps = (training_queries // batch_size) + 1

if len(test_ques) % batch_size == 0:
    validation_steps = len(test_ques) // batch_size
else:
    validation_steps = (len(test_ques) // batch_size) + 1


# Create the model


def basic_model(input_dims=512):
    inputs = Input(shape=(input_dims,), dtype="float32")
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(rate=0.0)(x)  # original 0.3
    x = Dense(64, activation='relu')(x)
    x = Dropout(rate=0.0)(x)  # original 0.3
    x = Dense(32, activation='relu')(x)
    x = Dropout(rate=0.0)(x)  # original 0.2
    y = Dense(1)(x)
    model = Model(inputs=inputs, outputs=y)

    return model


base_model = basic_model(input_dims=512)

inp1 = Input(shape=(512,), dtype='float32',
             name='Correct_Paragraph')  # Always some variation of True paragraph and Question
inp2 = Input(shape=(512,), dtype='float32',
             name='Original_Paragraph')  # Some variation of False + True paragraph and Question

y1 = base_model(inp1)
y2 = base_model(inp2)
y = Subtract()([y1, y2])
y_out = Activation("sigmoid")(y)

final_model = Model([inp1, inp2], y_out)
final_model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['binary_accuracy'])

#filepath = "ranknet_v1_{epoch:02d}_{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]

final_model.fit_generator(generator=training_generator, steps_per_epoch=training_steps,
                          epochs=1, validation_data=validation_generator, validation_steps=validation_steps)

final_model.save_weights("ranknet_v1_absolute_diff_model_weight_0_drop_1_top3.h5")