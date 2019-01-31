import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

with open(u'/home/hinton/bhargav/active_qa/bert-as-service/ms_data/eval1_unlabelled.tsv') as f:
    content = f.readlines()

id_list=[]
ques_list=[]
paras_list=[]
print("Reading File")
for i,x in enumerate(content):
    temp_lt = x.split("\t")
    id_list.append(temp_lt[0])
    ques_list.append(temp_lt[1])
    paras_list.append(temp_lt[2])
print("File has been read")
print("Reading TF module")
module_url = "/home/hinton/bhargav/active_qa/bert-as-service/ms_data/use_module/96e8f1d3d4d90ce86b2db128249eb8143a91db73"
embd = hub.Module(module_url)

print("Id list len={}\nQuestion Len={}\nParas Len={}".format(len(id_list),len(ques_list),len(paras_list)))

ques_ph = tf.placeholder(dtype=tf.float32, shape=[None,10,512],name="Questions")
paras_ph = tf.placeholder(dtype=tf.float32, shape=[None,10,512],name="Paragraphs")
drop_prob = tf.placeholder_with_default(0.0, shape=None,name='Dropout_prob')  # Default No dropout
q_ph = tf.placeholder(dtype=tf.string, shape=(None))
p_ph = tf.placeholder(dtype=tf.string, shape=(None))

# Point-wise FF layer


def point_wise_ff(ques_ph1, paras_ph1, drop_prob1, name_scope):

    with tf.name_scope(name= name_scope):
        o1 = tf.layers.dense(tf.concat([ques_ph1, paras_ph1], axis=2), 256, activation='relu')
        o1_dp = tf.layers.dropout(inputs=o1, rate=drop_prob1)
        o11 = tf.layers.dense(o1_dp, 128, activation="relu")
        o11_dp = tf.layers.dropout(inputs=o11, rate=drop_prob1)
        o111 = tf.layers.dense(o11_dp, 128, activation="relu")
        o111_dp = tf.layers.dropout(o111, rate=drop_prob1)
        out1 = tf.layers.dense(o111_dp, 1, activation=None)
    return out1


y_out1 = point_wise_ff(ques_ph, paras_ph, drop_prob, name_scope="Head1")  # Batch_size,10,1
y_out2 = point_wise_ff(ques_ph, paras_ph, drop_prob, name_scope="Head2")
y_out3 = point_wise_ff(ques_ph, paras_ph, drop_prob, name_scope="Head3")  # Batch_size,10,1
y_out4 = point_wise_ff(ques_ph, paras_ph, drop_prob, name_scope="Head4")

y_out = tf.nn.softmax(y_out1 + y_out2 + y_out3 + y_out4, axis=1)  # Batch_size,10,1

q_vec = embd(q_ph)
p_vec = embd(p_ph)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

init = tf.global_variables_initializer()
t_init = tf.tables_initializer()

# Checkpoint Saver
saver = tf.train.Saver(max_to_keep=10)

import csv
tsvfile=open('answer.tsv','w')
writer=csv.writer(tsvfile, delimiter='\t')

with tf.Session() as sess:
    sess.run([init,t_init])
    saver.restore(sess=sess, save_path="./models/msaic_use_2heads_600.ckpt")
    for i in range(0,len(id_list),10):
        print(i)
        temp_id = id_list[i]
        temp_ques = ques_list[i: i+10]
        temp_paras = paras_list[i: i+10]

        ques_vec = sess.run(q_vec, feed_dict={q_ph: temp_ques})
        paras_vec = sess.run(p_vec, feed_dict={p_ph: temp_paras})

        q_v = np.reshape(ques_vec, newshape=(-1, 10, ques_vec.shape[-1]))
        p_v = np.reshape(paras_vec, newshape=(-1, 10, paras_vec.shape[-1]))

        out = sess.run(y_out, feed_dict={ques_ph: q_v, paras_ph: p_v, drop_prob: 0.0})
        out_lt = out.reshape(-1, ).tolist()
        ip_lt = [temp_id] + out_lt
        writer.writerow(ip_lt)
tsvfile.close()