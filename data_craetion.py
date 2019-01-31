import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

with open("/home/hinton/bhargav/active_qa/bert-as-service/ms_data/data_pairwise_1.tsv", 'r') as f:
    content = f.readlines()


ques_list = []
#paras_list = []
#answers_list = []

print("Reading file content..")
for i, x in enumerate(content):
    temp_lt = x.split("\t")
    ques_list.append(temp_lt[1])
    #paras_list.append(temp_lt[1])

del(content)

print("Reading TF module")
module_url = "/home/hinton/bhargav/active_qa/bert-as-service/ms_data/use_module/96e8f1d3d4d90ce86b2db128249eb8143a91db73"
embd = hub.Module(module_url)

q_ph = tf.placeholder(dtype=tf.string, shape=(None))
q_vec = embd(q_ph)

init = tf.global_variables_initializer()
t_init = tf.tables_initializer()

batch = 2000
qv = np.array([], dtype=np.float32)
steps = len(ques_list) // batch + 1
print("Total steps = {}".format(steps))
with tf.Session() as sess:
    sess.run([init, t_init])
    for i in range(steps):
        i1 = i * batch
        print(i)
        if i == 0:
            qv = sess.run(q_vec, feed_dict={q_ph: ques_list[i1: i1 + batch]})
            qv = np.reshape(qv, newshape=(-1,10,512))
        elif i == steps-1:
            temp = sess.run(q_vec, feed_dict={q_ph: ques_list[i1: len(ques_list)]})
            temp = np.reshape(temp, newshape=(-1,10,512))
            qv = np.vstack((qv, temp))
        else:
            temp = sess.run(q_vec, feed_dict={q_ph: ques_list[i1: i1 + batch]})
            temp = np.reshape(temp, newshape=(-1, 10, 512))
            qv = np.vstack((qv, temp))
np.save("Paragraphs_use.npy", qv)
