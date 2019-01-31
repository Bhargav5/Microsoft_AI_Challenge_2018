import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

with open("/home/hinton/bhargav/active_qa/bert-as-service/ms_data/data_pairwise_1.tsv", 'r') as f:
    content = f.readlines()


ques_list = []
paras_list = []
answers_list = []

print("Reading file content..")
for i, x in enumerate(content):
    temp_lt = x.split("\t")
    ques_list.append(temp_lt[0])
    paras_list.append(temp_lt[1])
    answers_list.append(float(temp_lt[2]))

answers1_list = [answers_list[i:i+10] for i in range(0, len(answers_list), 10)]

print("Questions = {}".format(len(ques_list)))
print("Paragraphs = {}".format(len(paras_list)))
print("Answers = {}".format(len(answers1_list)))

# Training validation split
'''
Total unique Queries : 524188
Training: 4,50,000
Validation: 74,188
'''
training_queries = 450000

train_ques_list = ques_list[0: 10 * training_queries]
test_ques_list = ques_list[10 * training_queries:]

train_paras_list = paras_list[: 10 * training_queries]
test_paras_list = paras_list[10 * training_queries:]

train_ans_list = answers1_list[: training_queries]
test_ans_list = answers1_list[training_queries:]

print("Training Questions = {}".format(len(train_ques_list)))
print("Training paragraphs = {}".format(len(train_paras_list)))
print("Training Answers = {}".format(len(train_ans_list)))

print("Reading TF module")
module_url = "/home/hinton/bhargav/active_qa/bert-as-service/ms_data/use_module/96e8f1d3d4d90ce86b2db128249eb8143a91db73"
embd = hub.Module(module_url)


# Validation
def calculate_validation_acc(batch_test_ans_list, preds):
    true_vals = np.array(batch_test_ans_list)
    arg_true_vals = np.argmax(true_vals, axis=1)
    arg_preds = np.argmax(preds, axis=1)
    acc = np.mean(arg_preds == arg_true_vals)

    return acc

heads = 8
# Define placeholders
ques_ph = tf.placeholder(dtype=tf.float32, shape=[None, 10, 512], name="Questions")
paras_ph = tf.placeholder(dtype=tf.float32, shape=[None, 10, 512], name="Paragraphs")
ans_ph = tf.placeholder(dtype=tf.float32, shape=[None,10], name="Answers")
drop_prob = tf.placeholder_with_default(0.0, shape=None, name='Dropout_prob')  # Default No dropout
q_ph = tf.placeholder(dtype=tf.string, shape=(None))
p_ph = tf.placeholder(dtype=tf.string, shape=(None))
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
#out1_shape = tf.shape(out1)
#out2 = tf.reshape(tensor=out1, shape=(out1_shape[0], out1_shape[-1]))  # Shape: [batch, 10 * heads]
out2 = tf.layers.dropout(out1, rate=drop_prob, training=is_training_ph)
final_out = tf.layers.dense(out1, 10, activation=None)  # Shape: [batch, 10]

y_out = tf.nn.softmax(final_out, axis=1)

# Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ans_ph, logits=final_out))
train_op = optimizer.minimize(loss)

q_vec = embd(q_ph)
p_vec = embd(p_ph)

init = tf.global_variables_initializer()
t_init = tf.tables_initializer()

# Checkpoint Saver
saver = tf.train.Saver(max_to_keep=30)

epochs = 30
batch_size = 200  # Number of Unique Queries

if training_queries % batch_size == 0:
    steps = int(training_queries / batch_size)
else:
    steps = int(training_queries / batch_size) + 1

if len(test_ans_list) % batch_size == 0:
    val_steps = int(len(test_ans_list) / batch_size)
else:
    val_steps = int(len(test_ans_list) / batch_size) + 1
print("Validation Steps={}".format(val_steps))

validation_freq = 4  # Num of validations per epoch
validation_at = int(steps / validation_freq)
validation_acc = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

print("Training Begins ...")

with tf.Session(config=config) as sess:
    sess.run([init, t_init])
    for i in range(epochs):
        for j in range(steps):
            j1 = j * 10 * batch_size  # Pointer to fetch data from questions and paragraph lists
            j2 = j * batch_size  # Pointer to fetch data from answers list
            if j1 + 10 * batch_size <= len(train_ques_list):
                q_lt = train_ques_list[j1: j1 + 10 * batch_size]
                p_lt = train_paras_list[j1: j1 + 10 * batch_size]
                a_lt = train_ans_list[j2: j2 + batch_size]
            else:
                q_lt = train_ques_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]
                p_lt = train_paras_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]
                a_lt = train_ans_list[training_queries - batch_size: training_queries]

            q_v = sess.run(q_vec, feed_dict={q_ph: q_lt})  # Shape: [batch * 10, 512]
            p_v = sess.run(p_vec, feed_dict={p_ph: p_lt})  # Shape: [batch * 10, 512]
            a_v = np.array(a_lt).reshape((len(a_lt), 10))

            q_v1 = np.reshape(q_v, newshape=(-1, 10, 512))
            p_v1 = np.reshape(p_v, newshape=(-1, 10, 512))

            train_loss, _ = sess.run([loss, train_op], feed_dict={
                ques_ph: q_v1, paras_ph: p_v1, ans_ph: a_v, drop_prob: 0.4, is_training_ph: True
            })

            if j == 0:
                print("Questions = {}".format(q_lt[:10]))
                print("Paragraphs = {}".format(p_lt[:10]))
                print("Answers = {}".format(a_lt[:1]))
                prds = sess.run(y_out, feed_dict={
                    ques_ph: q_v1[:1], paras_ph: p_v1[:1], drop_prob: 0.0, is_training_ph: False
                })
                print("Predictions = {}".format(prds))

            print("Epoch: {}, step: {}, Loss: {}, Total_steps: {}".format(i, j, train_loss, steps))

            if j % 200 == 0:
                saver.save(sess, "./models/msaic_use_ce_attention_{}_{}.ckpt".format(i, j))
            if j % validation_at == 0:
                val_acc = []
                print("Validation Begins...")
                for k in range(val_steps):
                    k1 = k * 10 * batch_size
                    k2 = k * batch_size
                    if k1 + 10 * batch_size <= len(test_ques_list):
                        q_lt = test_ques_list[k1: k1 + 10 * batch_size]
                        p_lt = test_paras_list[k1: k1 + 10 * batch_size]
                        a_lt = test_ans_list[k2: k2 + batch_size]
                    else:
                        q_lt = test_ques_list[len(test_ques_list) - (10 * batch_size): len(test_ques_list)]
                        p_lt = test_paras_list[len(test_paras_list) - (10 * batch_size): len(test_paras_list)]
                        a_lt = test_ans_list[len(test_ans_list) - batch_size: len(test_ans_list)]

                    q_v = sess.run(q_vec, feed_dict={q_ph: q_lt})  # Shape: [batch * 10, 512]
                    p_v = sess.run(p_vec, feed_dict={p_ph: p_lt})  # Shape: [batch * 10, 512]
                    a_v = np.array(a_lt).reshape((len(a_lt), 10))

                    q_v1 = np.reshape(q_v, newshape=(-1, 10, 512))
                    p_v1 = np.reshape(p_v, newshape=(-1, 10, 512))

                    val_preds = sess.run(y_out, feed_dict={
                        ques_ph: q_v1, paras_ph: p_v1, is_training_ph: False, drop_prob: 0.0
                    })
                    batch_acc = calculate_validation_acc(a_v, preds=val_preds)
                    print("Epoch: {}, step: {}, Batch_Acc: {}, Total_steps: {}".format(i, k, batch_acc, val_steps))
                    val_acc.append(batch_acc)
                print("Validation Accuracy = {}".format(np.mean(val_acc)))










