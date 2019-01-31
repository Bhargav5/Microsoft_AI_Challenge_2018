import tensorflow as tf
import numpy as np

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

print("Reading Questions")
questions = np.load("Questions_use.npy")
print("Reading Paragraphs")
paragraphs = np.load("Paragraphs_use.npy")

training_queries = 450000

train_ques = questions[: training_queries]
train_paras = paragraphs[: training_queries]
train_ans = answers1_list[: training_queries]

test_ques = questions[training_queries:]
test_paras = paragraphs[training_queries:]
test_ans = answers1_list[training_queries:]

print("Train Questions = {} Train Paragraphs = {} Train Answers = {}".format(train_ques.shape, train_paras.shape,
                                                                             len(train_ans)))
print("Test Questions = {} Test Paragraphs = {} Test Answers = {}".format(test_ques.shape, test_paras.shape,
                                                                          len(test_ans)))


def calculate_validation_acc (true_vals, preds):
    #true_vals = np.array(batch_test_ans_list)
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
ans_ph = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="Answers")
drop_prob = tf.placeholder_with_default(0.0, shape=None, name='Dropout_prob')  # Default No dropout
is_training_ph = tf.placeholder(dtype=tf.bool, shape=None)

# Network architecture

Q = tf.layers.dense(ques_ph, units=512, activation=tf.nn.relu)  # shape: [batch, 10, 512]
P = tf.layers.dense(paras_ph, units=512, activation=tf.nn.relu)  # shape: [batch, 10, 512]

Q_split = tf.concat(tf.split(Q, heads, axis=2), axis=0)  # shape: [batch * heads, 10, 512 / heads]
P_split = tf.concat(tf.split(P, heads, axis=2), axis=0)  # shape: [batch * heads, 10, 512 / heads]

d = 512 // heads
assert d == Q_split.shape[-1] == P_split.shape[-1]

out = tf.matmul(Q_split,
                tf.transpose(P_split, [0, 2, 1]))  # (10, 512 / heads) x (512 / heads , 10) = [batch * heads, 10, 10]
out = out / tf.sqrt(tf.cast(d, tf.float32))

out = tf.reduce_mean(out, axis=1)  # Shape:[batch * heads, 10]
out1 = tf.concat(tf.split(out, num_or_size_splits=heads, axis=0), axis=1)  # Shape:[batch, 10 * heads]

#out2 = tf.layers.dropout(out1, rate=drop_prob, training=is_training_ph)
out2 = tf.reshape(out1, (-1, 10, heads))
final_out = tf.reduce_mean(out2, axis=2)

y_out = tf.nn.softmax(final_out, axis=1)

# Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ans_ph, logits=final_out))
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
t_init = tf.tables_initializer()

# Checkpoint Saver
saver = tf.train.Saver(max_to_keep=30)

epochs = 100
batch_size = 200  # Number of Unique Queries

if training_queries % batch_size == 0:
    steps = int(training_queries / batch_size)
else:
    steps = int(training_queries / batch_size) + 1

if len(test_ans) % batch_size == 0:
    val_steps = int(len(test_ans) / batch_size)
else:
    val_steps = int(len(test_ans) / batch_size) + 1
print("Validation Steps={}".format(val_steps))

validation_freq = 2  # Num of validations per epoch
validation_at = int(steps / validation_freq)
# validation_acc = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
best_validation_acc = 0
print("Training Begins ...")
with tf.Session(config=config) as sess:
    sess.run([init, t_init])
    for i in range(epochs):
        epoch_loss = []
        for j in range(steps):
            j1 = j * batch_size  # Pointer to fetch data from questions and paragraph lists

            if j1 + batch_size <= training_queries:
                q_v = train_ques[j1: j1 + batch_size]
                p_v = train_paras[j1: j1 + batch_size]
                a_lt = train_ans[j1: j1 + batch_size]
            else:
                q_v = train_ques[j1: training_queries]
                p_v = train_paras[j1: training_queries]
                a_lt = train_ans[j1: training_queries]

            a_v = np.array(a_lt, dtype=np.float32)
            a_v = np.reshape(a_v, newshape=(-1, 10))

            train_loss, _ = sess.run([loss, train_op], feed_dict={
                ques_ph: q_v, paras_ph: p_v, ans_ph: a_v, drop_prob: 0.4, is_training_ph: True
            })
            epoch_loss.append(train_loss)

            #print("Epoch: {}, step: {}, Loss: {}, Total_steps: {}".format(i, j, train_loss, steps))

            #if j % 200 == 0:
                #saver.save(sess, "./models/msaic_use_ce_attention__avg_{}_{}.ckpt".format(i, j))
            if j % validation_at == 0:
                val_acc = []
                print("Validation Begins...")
                for k in range(val_steps):
                    k1 = k * batch_size

                    if k1 + batch_size <= training_queries:
                        q_v = train_ques[k1: k1 + batch_size]
                        p_v = train_paras[k1: k1 + batch_size]
                        a_lt = train_ans[k1: k1 + batch_size]
                    else:
                        q_v = train_ques[k1: training_queries]
                        p_v = train_paras[k1: training_queries]
                        a_lt = train_ans[k1: training_queries]

                    a_v = np.array(a_lt, dtype=np.float32)
                    a_v = np.reshape(a_v, newshape=(-1, 10))

                    val_preds = sess.run(y_out, feed_dict={
                        ques_ph: q_v, paras_ph: p_v, is_training_ph: False, drop_prob: 0.0
                    })
                    batch_acc = calculate_validation_acc(a_v, preds=val_preds)
                    #print("Epoch: {}, step: {}, Batch_Acc: {}, Total_steps: {}".format(i, k, batch_acc, val_steps))
                    val_acc.append(batch_acc)
                current_val_acc = np.mean(val_acc)
                print("Validation Accuracy = {}, Best Validation Acc = {}".format(current_val_acc, best_validation_acc))
                if current_val_acc > best_validation_acc:
                    best_validation_acc = current_val_acc
                    saver.save(sess, "./models/msaic_use_ce_attention__avg_{}_{}.ckpt".format(i, j))
        print("Epoch = {}, Epoch Loss = {}".format(i, np.mean(epoch_loss)))



