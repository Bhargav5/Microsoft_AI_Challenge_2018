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

# Define Placeholders
ques_ph = tf.placeholder(dtype=tf.float32, shape=[None,512],name="Questions")
pos_paras_ph = tf.placeholder(dtype=tf.float32, shape=[None,512],name="Pos_Paragraphs")
neg_paras_ph = tf.placeholder(dtype=tf.float32, shape=[None,512], name="Neg_Paragraphs")
out_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Output")
drop_prob = tf.placeholder_with_default(0.0, shape=None,name='Dropout_prob')  # Default No dropout
q_ph = tf.placeholder(dtype=tf.string, shape=(None))
p_ph = tf.placeholder(dtype=tf.string, shape=(None))
    

pos_point_mul = ques_ph * pos_paras_ph
pos_addition = ques_ph + pos_paras_ph
pos_abs_diff = tf.abs(ques_ph - pos_paras_ph)

neg_point_mul = ques_ph * neg_paras_ph
neg_addition = ques_ph + neg_paras_ph
neg_abs_diff = tf.abs(ques_ph - neg_paras_ph)


def classification_layer(addition, point_mul, abs_diff, drop_prob):

    m1 = tf.layers.dense(point_mul, 256, activation='relu')
    m1 = tf.layers.dropout(m1, rate=drop_prob)
    m1 = tf.layers.dense(m1, 16)

    d1 = tf.layers.dense(abs_diff, 256, activation='relu')
    d1 = tf.layers.dropout(d1, rate=drop_prob)
    d1 = tf.layers.dense(d1, 16)

    a1 = tf.layers.dense(addition, 256, activation='relu')
    a1 = tf.layers.dropout(a1, rate=drop_prob)
    a1 = tf.layers.dense(a1, 16)

    conc = tf.concat([m1, a1, d1], axis=1)
    conc = tf.layers.dropout(conc, rate=drop_prob)
    y_out = tf.layers.dense(conc, 1, activation=None)

    return y_out


print(pos_out == neg_out)
#print([x.name for x in tf.global_variables()])
print(pos_out)
print(neg_out)

out_diff = pos_out - neg_out
eps = 1e-6
clipped_out_diff = tf.clip_by_value(out_diff, eps, 1-eps)
loss = tf.reduce_mean(-tf.reduce_sum(out_ph * tf.log(clipped_out_diff), axis=1))

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

q_vec = embd(q_ph)
p_vec = embd(p_ph)

init = tf.global_variables_initializer()
t_init = tf.tables_initializer()

# Checkpoint Saver
saver = tf.train.Saver(max_to_keep=30)

epochs = 3
batch_size = 200  # Number of Unique Queries

if len(ques_list) % batch_size == 0:
    steps = int(len(ques_list) / (10 * batch_size))
else:
    steps = int(len(ques_list) / (10 * batch_size)) + 1


config = tf.ConfigProto()
config.gpu_options.allow_growth=True

print("Training Begins ...")

with tf.Session(config=config) as sess:
    sess.run([init, t_init])
    for i in range(epochs):
        # shuffle the data
        lt_temp = list(zip(ques_list, pos_paras_list, neg_paras_list))
        random.shuffle(lt_temp)
        train_ques_list, train_pos_paras_list, train_neg_paras_list = zip(*lt_temp)
        for j in range(steps):
            j1 = j * 10 * batch_size  # Pointer to fetch data from questions and paragraph lists

            if j1 + 10 * batch_size <= len(train_ques_list):
                q_lt = train_ques_list[j1: j1 + 10 * batch_size]
                pos_p_lt = train_pos_paras_list[j1: j1 + 10 * batch_size]
                neg_p_lt = train_neg_paras_list[j1: j1 + 10 * batch_size]
            else:
                q_lt = train_ques_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]
                pos_p_lt = train_pos_paras_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]
                neg_p_lt = train_neg_paras_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]

            q_v = sess.run(q_vec, feed_dict={q_ph: q_lt})
            pos_p_v = sess.run(p_vec, feed_dict={p_ph: pos_p_lt})
            neg_p_v = sess.run(p_vec, feed_dict={p_ph: neg_p_lt})
            a_v = np.ones(shape=(10 * batch_size, 1), dtype=np.float32)

            if j==0:
                diff = sess.run(loss, feed_dict={
                ques_ph: q_v[:3], pos_paras_ph: pos_p_v[:3], neg_paras_ph:neg_p_v[:3], out_ph: a_v[:3], drop_prob: 0.4})
                print(diff)



            train_loss, _ = sess.run([loss, train_op], feed_dict={
                ques_ph: q_v, pos_paras_ph: pos_p_v, neg_paras_ph:neg_p_v, out_ph: a_v, drop_prob: 0.4
            })

            print("Epoch: {}, step: {}, Loss: {}, Total_steps: {}".format(i, j, train_loss, steps))

            if j % 400 == 0:
                saver.save(sess, "./models/msaic_use_ce_pw_shuffle_drop_40_{}_{}.ckpt".format(i, j))


