import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

with open(u'/home/hinton/bhargav/active_qa/bert-as-service/ms_data/eval1_unlabelled.tsv') as f:
    content = f.readlines()

id_list=[]

print("Reading File")
for i,x in enumerate(content):
    temp_lt = x.split("\t")
    id_list.append(temp_lt[0])

id_list1 = [id_list[i] for i in range(0,len(id_list),10)]
ques_vec = np.load("Validation_Questions_use.npy")
paras_vec = np.load("Validation_Paragraphs_use.npy")
print("File has been read")
print("Id list len={}\nQuestion Len={}\nParas Len={}".format(len(id_list1),ques_vec.shape,paras_vec.shape))

heads = 8
# Define placeholders
ques_ph = tf.placeholder(dtype=tf.float32, shape=[None, 10, 512], name="Questions")
paras_ph = tf.placeholder(dtype=tf.float32, shape=[None, 10, 512], name="Paragraphs")
ans_ph = tf.placeholder(dtype=tf.float32, shape=[None,10], name="Answers")
drop_prob = tf.placeholder_with_default(0.0, shape=None, name='Dropout_prob')  # Default No dropout
is_training_ph = tf.placeholder(dtype=tf.bool, shape=None)


# Network architecture

Q = tf.layers.dense(ques_ph, units=512, activation=tf.nn.relu)  # shape: [batch, 10, 512]
P = tf.layers.dense(paras_ph, units=512, activation=tf.nn.relu)  # shape: [batch, 10, 512]

Q_split = tf.concat(tf.split(Q, heads, axis=2), axis=0)  # shape: [batch * heads, 10, 512 / heads]
P_split = tf.concat(tf.split(P, heads, axis=2), axis=0)  # shape: [batch * heads, 10, 512 / heads]

d = 512 // heads
assert d == Q_split.shape[-1] == P_split.shape[-1]

out = tf.matmul(Q_split, tf.transpose(P_split, [0, 2, 1]))  # (10, 512 / heads) x (512 / heads , 10) = [batch * heads, 10, 10]
out = out / tf.sqrt(tf.cast(d, tf.float32))

out = tf.reduce_mean(out, axis=1)  # Shape:[batch * heads, 10]
out1 = tf.concat(tf.split(out, num_or_size_splits=heads, axis=0), axis=1)  # Shape:[batch, 10 * heads]

out2 = tf.layers.dropout(out1, rate=drop_prob, training=is_training_ph)
final_out = tf.layers.dense(out1, 10, activation=None)  # Shape: [batch, 10]

y_out = tf.nn.softmax(final_out, axis=1)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ans_ph, logits=final_out))
train_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()
t_init = tf.tables_initializer()

# Checkpoint Saver
saver = tf.train.Saver(max_to_keep=30)

import csv
tsvfile=open('answer.tsv','w')
writer=csv.writer(tsvfile, delimiter='\t')

with tf.Session() as sess:
    sess.run([init,t_init])
    saver.restore(sess=sess, save_path="./models/msaic_use_ce_attention_29_2200.ckpt")
    for i in range(0,len(id_list1)):
        print(i)
        temp_id = id_list1[i]

        ques_vec1 = ques_vec[i]
        paras_vec1 = paras_vec[i]

        ques_vec1 = np.reshape(ques_vec1, newshape=(-1, 10, 512))
        paras_vec1 = np.reshape(paras_vec1, newshape=(-1, 10, 512))

        out = sess.run(y_out, feed_dict={ques_ph: ques_vec1, paras_ph: paras_vec1, drop_prob: 0.0, is_training_ph: False})
        out_lt = out.reshape(-1,).tolist()
        ip_lt = [temp_id] + out_lt
        writer.writerow(ip_lt)
tsvfile.close()