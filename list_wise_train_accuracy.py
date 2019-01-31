import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

with open("/home/hinton/bhargav/active_qa/bert-as-service/ms_data/data_pairwise_1.tsv", 'r') as f:
    content = f.readlines()


ans_list = []

print("Reading file content..")
for i, x in enumerate(content):
    temp_lt = x.split("\t")
    ans_list.append(float(temp_lt[2]))
answers1_list = [ans_list[i:i+10] for i in range(0, len(ans_list), 10)]
del(content)
del(ans_list)

id_list1 = list(range(0, len(answers1_list)))

ques_vec = np.load("Questions_use.npy")
paras_vec = np.load("Paragraphs_use.npy")
print("File has been read")
print("Id list len={}\nQuestion Len={}\nParas Len={}".format(len(id_list1),ques_vec.shape,paras_vec.shape))


def calculate_validation_acc (batch_test_ans_list, preds):
    true_vals = np.array(batch_test_ans_list)
    arg_true_vals = np.argmax(true_vals, axis=1)
    arg_preds = np.argsort(preds, axis=1)
    last = np.array([x[-1] for x in arg_preds], dtype=np.float32)
    last1 = np.array([x[-2] for x in arg_preds], dtype=np.float32)
    last2 = np.array([x[-3] for x in arg_preds], dtype=np.float32)
    acc = np.mean(last == arg_true_vals)
    acc2 = np.mean(last1 == arg_true_vals)
    acc3 = np.mean(last2 == arg_true_vals)
    return acc + 0.5 * acc2 +  0.33 * acc3


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
tsvfile=open('train_answer.tsv','w')
writer=csv.writer(tsvfile, delimiter='\t')
validation_acc = []
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
        #validation_acc.append(calculate_validation_acc(answers1_list[i], out))
    #print("Average Accuracy = {}".format(np.mean(validation_acc)))
tsvfile.close()