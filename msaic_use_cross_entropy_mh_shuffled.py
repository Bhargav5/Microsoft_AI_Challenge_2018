import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import random

with open("/home/hinton/bhargav/active_qa/bert-as-service/ms_data/data.tsv", 'r') as f:
    content = f.readlines()

ques_list = []  # List of all the questions, including duplicate
paras_list = []  # List of all the paragraphs
ans_list = [] # list of all the answers
print("Start Reading file...")
for i, x in enumerate(content):
    temp_lt = x.split("\t")
    if float(temp_lt[3]) == 1.0:
        for k in range(5):
            ques_list.append(temp_lt[1])
            paras_list.append(temp_lt[2])
            ans_list.append(float(temp_lt[3]))
    else:
        ques_list.append(temp_lt[1])
        paras_list.append(temp_lt[2])
        ans_list.append(float(temp_lt[3]))

#ans_list1 = [ans_list[i:i+10] for i in range(0, len(ans_list), 10)]

print("Question Len={}".format(len(ques_list)))
print("Para Len={}".format(len(paras_list)))
print("Ans Len={}".format(len(ans_list)))

training_queries = int(0.095 * len(ques_list))
'''
Total unique queries = 524188
Training --> 5,14,188
Validation --> 10,000
'''
train_ques_list = ques_list[: 10 * training_queries]
test_ques_list = ques_list[10 * training_queries:]

train_paras_list = paras_list[: 10 * training_queries]
test_paras_list = paras_list[10 * training_queries:]

train_ans_list = ans_list[: 10 * training_queries]
test_ans_list = ans_list[10 * training_queries:]

print("Training Questions = {}, Testing Questions = {}".format(len(train_ques_list), len(test_ques_list)))
print("Training Paragraphs = {}. Testing Paragraphs = {}".format(len(train_paras_list), len(test_paras_list)))

print("Reading TF module")
module_url = "/home/hinton/bhargav/active_qa/bert-as-service/ms_data/use_module/96e8f1d3d4d90ce86b2db128249eb8143a91db73"
embd = hub.Module(module_url)


def testing_reward_function(answers, predictions):
    """
    Calculates MRR (Mean Reciprocal Rank) for predictions
    :param answers: shape: Batch,10
    :param predictions: shape: Batch,10
    :return: shape: Batch,1
    """
    rewards = []
    args = np.argmax(answers, axis=1)
    y_pred = predictions
    for i in range(0, len(y_pred)):
        lt1 = y_pred[i] >= y_pred[i][args[i]]
        s = np.sum(lt1)
        rewards.append(1.0 / s)
    return rewards


ques_ph = tf.placeholder(dtype=tf.float32, shape=[None,512],name="Questions")
paras_ph = tf.placeholder(dtype=tf.float32, shape=[None,512],name="Paragraphs")
out_ph = tf.placeholder(dtype=tf.float32, shape=[None,1], name="Output")
drop_prob = tf.placeholder_with_default(0.0, shape=None,name='Dropout_prob')  # Default No dropout
q_ph = tf.placeholder(dtype=tf.string, shape=(None))
p_ph = tf.placeholder(dtype=tf.string, shape=(None))

point_mul = ques_ph * paras_ph
addition = ques_ph + paras_ph
abs_diff = tf.abs(ques_ph - paras_ph)

'''
def classification_layer(ques, paras, point_mul, abs_diff, drop_prob):

    m1 = tf.layers.dense(point_mul, 256, activation='relu')
    m1 = tf.layers.dropout(m1, rate=drop_prob)
    m1 = tf.layers.dense(m1, 16, activation='relu')

    d1 = tf.layers.dense(abs_diff, 256, activation='relu')
    d1 = tf.layers.dropout(d1, rate=drop_prob)
    d1 = tf.layers.dense(d1, 16, activation='relu')

    q1 = tf.layers.dense(ques, 256, activation='relu')
    q1 = tf.layers.dropout(q1, rate=drop_prob)
    q1 = tf.layers.dense(q1, 16, activation='relu')

    p1 = tf.layers.dense(paras, 256, activation='relu')
    p1 = tf.layers.dropout(p1, rate=drop_prob)
    p1 = tf.layers.dense(p1, 16, activation='relu')

    conc = tf.concat([q1, p1, d1, m1], axis=1)
    conc = tf.layers.dropout(conc, rate=drop_prob)
    y_out = tf.layers.dense(conc, 1, activation=None)

    return y_out
'''


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


out1 = classification_layer(addition, point_mul, abs_diff, drop_prob)
out2 = classification_layer(addition, point_mul, abs_diff, drop_prob)
out3 = classification_layer(addition, point_mul, abs_diff, drop_prob)
out4 = classification_layer(addition, point_mul, abs_diff, drop_prob)

con = tf.concat([out1, out2, out3, out4], axis=1)
con = tf.layers.dropout(con, rate=drop_prob)
final_out = tf.layers.dense(con, 1, activation=None)


optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=out_ph, logits=final_out))
train_op = optimizer.minimize(loss)

q_vec = embd(q_ph)
p_vec = embd(p_ph)

init = tf.global_variables_initializer()
t_init = tf.tables_initializer()

# Checkpoint Saver
saver = tf.train.Saver(max_to_keep=30)

epochs = 3
batch_size = 200  # Number of Unique Queries

if training_queries % batch_size == 0:
    steps = int(training_queries / batch_size)
else:
    steps = int(training_queries / batch_size) + 1

if len(test_ans_list) % batch_size == 0:
    val_steps = int(len(test_ans_list) / (10 * batch_size))
else:
    val_steps = int(len(test_ans_list) / (10 * batch_size)) + 1
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
        # shuffle the data
        lt_temp = list(zip(train_ques_list, train_paras_list, train_ans_list))
        random.shuffle(lt_temp)
        train_ques_list, train_paras_list, train_ans_list = zip(*lt_temp)
        for j in range(steps):
            j1 = j * 10 * batch_size  # Pointer to fetch data from questions and paragraph lists
            j2 = j * batch_size  # Pointer to fetch data from answers list
            if j1 + 10 * batch_size <= len(train_ques_list):
                q_lt = train_ques_list[j1: j1 + 10 * batch_size]
                p_lt = train_paras_list[j1: j1 + 10 * batch_size]
                a_lt = train_ans_list[j1: j1 + 10 * batch_size]
            else:
                q_lt = train_ques_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]
                p_lt = train_paras_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]
                a_lt = train_ans_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]

            q_v = sess.run(q_vec, feed_dict={q_ph: q_lt})
            p_v = sess.run(p_vec, feed_dict={p_ph: p_lt})
            a_v = np.reshape(np.array(a_lt), newshape=(-1, 1))

            train_loss, _ = sess.run([loss, train_op], feed_dict={
                ques_ph: q_v, paras_ph: p_v, out_ph: a_v, drop_prob: 0.4
            })

            print("Epoch: {}, step: {}, Loss: {}, Total_steps: {}".format(i, j, train_loss, steps))

            if j % 200 == 0:
                saver.save(sess, "./models/msaic_use_ce_mh_lr_10e4_shuffle_drop_40_{}_{}.ckpt".format(i, j))
            '''
            if j % validation_at == 0:
                print("Validation Begins..")
                val_reward=[]
                for k in range(val_steps):
                    print("Steps {} of {}".format(k,val_steps))
                    k1 = k * 10 * batch_size  # Pointer to fetch data from questions and paragraph lists

                    if k1 + 10 * batch_size <= len(test_ques_list):
                        q_lt = test_ques_list[k1: k1 + 10 * batch_size]
                        p_lt = test_paras_list[k1: k1 + 10 * batch_size]
                        a_lt = test_ans_list[k1: k1 + 10 * batch_size]
                    else:
                        q_lt = test_ques_list[len(test_ques_list) - (10 * batch_size): len(test_ques_list)]
                        p_lt = test_paras_list[len(test_ques_list) - (10 * batch_size): len(test_ques_list)]
                        a_lt = test_ans_list[len(test_ques_list) - (10 * batch_size): len(test_ques_list)]

                    q_v = sess.run(q_vec, feed_dict={q_ph: q_lt})
                    p_v = sess.run(p_vec, feed_dict={p_ph: p_lt})
                    a_v = np.reshape(np.array(a_lt), newshape=(-1, 1))

                    val_op = sess.run(final_out, feed_dict={
                        ques_ph: q_v, paras_ph: p_v, drop_prob: 0.0
                    })

                    val_reward.extend(testing_reward_function(answers=a_v, predictions=val_op))
                    if (k % 25 ==0):
                        print(testing_reward_function(answers=a_v, predictions=val_op)[:5], a_v[:5], val_op[:5])
                        #print(val_reward)
                    print(val_reward)
                avg_reward = np.mean(val_reward)
                print("Validation Acc = {}".format(avg_reward))

                if avg_reward > validation_acc:
                    validation_acc = avg_reward
                    saver.save(sess, "./models/msaic_use_ce_mh_shuffle_drop_40_best_{}_{}.ckpt".format(i,j))
            '''



