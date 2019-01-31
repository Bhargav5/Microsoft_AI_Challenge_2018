import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Subtract, Activation
from keras.models import Model, load_model
from keras.engine import Layer
import tensorflow_hub as hub
import tensorflow as tf
from keras.optimizers import Adam

# initialize session
sess = tf.Session()
K.set_session(sess)

with open("/home/hinton/bhargav/active_qa/bert-as-service/ms_data/train1.tsv", 'r') as f:
    content = f.readlines()
ques_list = []  # List of all the questions, including duplicate
paras_list = []  # List of all the paragraphs
ans_list = [] # list of all the answers
print("Start Reading file...")
for i, x in enumerate(content):
    temp_lt = x.split("\t")
    ques_list.append(temp_lt[0])
    paras_list.append(temp_lt[1])
    ans_list.append(float(temp_lt[2]))

training_len = int(len(ques_list) * 0.8)
train_ques = ques_list[:training_len]
train_para = paras_list[:training_len]
train_ans = ans_list[:training_len]
train_ans = np.array(train_ans, dtype=np.float16)
train_ans = train_ans.reshape(-1,1)

test_ques = ques_list[training_len:]
test_para = paras_list[training_len:]
test_ans = ans_list[training_len:]
test_ans = np.array(test_ans, dtype=np.float16)
test_ans = test_ans.reshape(-1, 1)

print("Train ans shape = {}, Test ans shape = {}".format(train_ans.shape, test_ans.shape))

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


in_ques = Input(shape=(1,), dtype='string')
in_para = Input(shape=(1,), dtype='string')
ques_embd = ElmoEmbeddingLayer()(in_ques)
para_embd = ElmoEmbeddingLayer()(in_para)
diff = Subtract()([ques_embd, para_embd])
x = Dense(256, activation='relu')(diff)
x = Dropout(rate=0.0)(x)  # original 0.3
x = Dense(64, activation='relu')(x)
x = Dropout(rate=0.0)(x)  # original 0.3
x = Dense(32, activation='relu')(x)
x = Dropout(rate=0.0)(x)  # original 0.2
y = Dense(1)(x)

model = Model(inputs=[in_ques, in_para], outputs=y)
optmizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optmizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

print(model.summary())

model.fit(x=[train_ques, train_para], y=train_ans, validation_data=([test_ques, test_para], test_ans), epochs=2, batch_size=8)

model.save_weights("fine_tuned_emlo_weights.h5")
