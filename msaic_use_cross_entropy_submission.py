import tensorflow_hub as hub
import tensorflow as tf

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

#ques=np.load("ms_data/test_question.npy")
#paras=np.load("ms_data/test_paras.npy")
print("Id list len={}\nQuestion Len={}\nParas Len={}".format(len(id_list),len(ques_list),len(paras_list)))
#ques_ph = tf.placeholder(tf.string,shape=(None))
#paras_ph = tf.placeholder(tf.string,shape=(None))
ques_ph = tf.placeholder(dtype=tf.float32, shape=[None,512],name="Questions")
paras_ph = tf.placeholder(dtype=tf.float32, shape=[None,512],name="Paragraphs")
out_ph = tf.placeholder(dtype=tf.float32, shape=[None,1], name="Output")
drop_prob = tf.placeholder_with_default(0.0, shape=None,name='Dropout_prob')  # Default No dropout
q_ph = tf.placeholder(dtype=tf.string, shape=(None))
p_ph = tf.placeholder(dtype=tf.string, shape=(None))

point_mul = ques_ph * paras_ph
addition = ques_ph + paras_ph
abs_diff = tf.abs(ques_ph - paras_ph)

m1 = tf.layers.dense(point_mul, 256, activation='relu')
m1 = tf.layers.dropout(m1, rate=drop_prob)
m1 = tf.layers.dense(m1, 16)

a1 = tf.layers.dense(addition, 256, activation='relu')
a1 = tf.layers.dropout(a1, rate=drop_prob)
a1 = tf.layers.dense(a1, 16)

d1 = tf.layers.dense(abs_diff, 256, activation='relu')
d1 = tf.layers.dropout(d1, rate=drop_prob)
d1 = tf.layers.dense(d1, 16)

conc = tf.concat([m1,a1,d1], axis=1)
conc = tf.layers.dropout(conc, rate=drop_prob)
y_out = tf.layers.dense(conc, units=1, activation=None)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=out_ph, logits=y_out))
train_op = optimizer.minimize(loss)

qv = embd(q_ph)
pv = embd(p_ph)

init = tf.global_variables_initializer()
t_init = tf.tables_initializer()
saver = tf.train.Saver(max_to_keep=10)

import csv
tsvfile=open('answer.tsv','w')
writer=csv.writer(tsvfile, delimiter='\t')

with tf.Session() as sess:
    sess.run([init,t_init])
    saver.restore(sess=sess, save_path="./models/msaic_use_ce_drop_30_1_500.ckpt")
    for i in range(0,len(id_list),10):
        print(i)
        temp_id = id_list[i]
        temp_ques = ques_list[i: i+10]
        temp_paras = paras_list[i: i+10]

        ques_vec = sess.run(qv, feed_dict={q_ph: temp_ques})
        paras_vec = sess.run(pv, feed_dict={p_ph: temp_paras})

        out = sess.run(y_out, feed_dict={ques_ph: ques_vec, paras_ph: paras_vec, drop_prob: 0.0})
        out_lt = out.reshape(-1,).tolist()
        ip_lt= [temp_id] + out_lt
        writer.writerow(ip_lt)
tsvfile.close()
