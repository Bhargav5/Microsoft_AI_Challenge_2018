import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

with open("/home/hinton/bhargav/active_qa/bert-as-service/ms_data/data.tsv", 'r') as f:
    content = f.readlines()

ques_list = []  # List of all the questions, including duplicate
paras_list = []  # List of all the paragraphs
ans_list = [] # list of all the answers
print("Start Reading file...")
for i, x in enumerate(content):
    temp_lt = x.split("\t")
    ques_list.append(temp_lt[1])
    paras_list.append(temp_lt[2])
    ans_list.append(temp_lt[3])


ans_list1 = [ans_list[i:i+10] for i in range(0, len(ans_list), 10)]

print("Question Len={}".format(len(ques_list)))
print("Para Len={}".format(len(paras_list)))
print("Ans Len={}".format(len(ans_list1)))

training_queries = 470000
'''
Total unique queries = 524188
Training --> 4,70,000
Validation --> 54,188
'''
train_ques_list = ques_list[: 10 * training_queries]
test_ques_list = ques_list[10 * training_queries:]

train_paras_list = paras_list[: 10 * training_queries]
test_paras_list = paras_list[10 * training_queries:]

train_ans_list = ans_list1[: training_queries]
test_ans_list = ans_list1[training_queries:]

print("Training Questions = {}, Testing Questions = {}".format(len(train_ques_list), len(test_ques_list)))
print("Training Paragraphs = {}. Testing Paragraphs = {}".format(len(train_paras_list), len(test_paras_list)))

print("Reading TF module")
module_url = "/home/hinton/bhargav/active_qa/bert-as-service/ms_data/use_module/96e8f1d3d4d90ce86b2db128249eb8143a91db73"
embd = hub.Module(module_url)

# Define tf variables


baseline = tf.Variable(-1.0, dtype=tf.float32, name="Baseline", trainable=False)
adv = tf.Variable(0.0, dtype=tf.float32, name="Advantage", trainable=False)

# Define reward functions


def training_reward_function(answers, predictions):
    """
    Calculates MRR (Mean Reciprocal Rank) for predictions
    :param answers: shape: Batch,10
    :param predictions: shape: Batch,10,1
    :return: shape: Batch,1,1
    """
    rewards = []
    args = np.argmax(answers, axis=1)
    y_pred = predictions
    for i in range(0, len(y_pred)):
        lt1 = y_pred[i] >= y_pred[i][args[i]]
        s = np.sum(lt1)
        if (s < 3.0):
            rewards.append(1.0)
        else:
            rewards.append(1.0 / s)
    rew = np.array(rewards)
    return rew.reshape(-1, 1, 1)


def testing_reward_function(answers, predictions):
    """
    Calculates MRR (Mean Reciprocal Rank) for predictions
    :param answers: shape: Batch,10
    :param predictions: shape: Batch,10,1
    :return: shape: Batch,1,1
    """
    rewards = []
    args = np.argmax(answers, axis=1)
    y_pred = predictions
    for i in range(0, len(y_pred)):
        lt1 = y_pred[i] >= y_pred[i][args[i]]
        s = np.sum(lt1)
        rewards.append(1.0 / s)
    rew = np.array(rewards)
    return rew.reshape(-1, 1, 1)

# Define Placeholders


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

epochs = 3
batch_size = 300  # Number of Unique Queries

if training_queries % batch_size == 0:
    steps = int(training_queries / batch_size)
else:
    steps = int(training_queries / batch_size) + 1
    
if len(test_ans_list) % batch_size == 0:
    val_steps = int(len(test_ans_list) / batch_size)
else:
    val_steps = int(len(test_ans_list) / batch_size) + 1
print("Validation Steps={}".format(val_steps))
validation_freq = 2  # Num of validations per epoch
validation_at = int(steps / validation_freq)
validation_acc = 0


def calculate_acc(labels, predictions,threshold=0.5):
    pred = (1 / (1 + np.exp(-predictions))) > threshold
    #pred.reshape(-1,)
    return np.mean(labels == pred)


print("Training Begins ...")

with tf. Session() as sess:
    sess.run([init, t_init])
    for i in range(epochs):
        for j in range(steps):
            j1 = j * 10 * batch_size  # Pointer to fetch data from questions and paragraph lists
            j2 = j * batch_size  # Pointer to fetch data from answers list
            if j1 + 10 * batch_size <= len(train_ques_list):
                q_lt = train_ques_list[j1: j1 + 10 * batch_size]
                p_lt = train_paras_list[j1: j1 + 10 * batch_size]
            else:
                q_lt = train_ques_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]
                p_lt = train_paras_list[len(train_ques_list) - (10 * batch_size): len(train_ques_list)]
            
            q_v = sess.run(q_vec, feed_dict={q_ph: q_lt})
            p_v = sess.run(p_vec, feed_dict={p_ph: p_lt})
            
            q_v = np.reshape(q_v, newshape=(-1, 10, q_v.shape[-1]))
            p_v = np.reshape(p_v, newshape=(-1, 10, p_v.shape[-1]))
            
            out = sess.run(y_out, feed_dict={ques_ph: q_v, paras_ph: p_v, drop_prob: 0.2})

            if j2 + batch_size <= len(train_ans_list):
                a_lt = train_ans_list[j2: j2 + batch_size]
            else:
                a_lt = train_ans_list[len(train_ans_list) - batch_size: len(train_ans_list)]

            reward_t = training_reward_function(answers=np.array(a_lt), predictions=out)  # np.array, shape: Batch,1,1
            adv = reward_t - baseline  # tf.Tensor, tf.shape: Batch,1,1

            baseline = 0.9 * baseline + 0.1 * np.mean(reward_t)  # Update baseline tf.Variable with exp mean
            loss = tf.reduce_mean(adv * tf.log(y_out))
            train_op = optimizer.minimize(-loss)

            training_loss, baseline_val, _ = sess.run([loss, baseline, train_op], feed_dict={
                ques_ph: q_v, paras_ph: p_v, drop_prob: 0.2
            })
            if j % 200 == 0:
                saver.save(sess=sess,save_path="./models/msaic_use_2heads_{}.ckpt".format(j))
                print("Model msaic_use_2heads_{}.ckpt saved".format(i))

            # Print Log
            print("Epoch: {}, step: {}, Baseline: {}, Loss: {}, Total_steps: {}".format(i, j, baseline_val, training_loss, steps))
            '''
            if j % validation_at == 0 and j !=0:
                print("Validation Begins..")
                val_reward=[]
                for k in range(val_steps):
                    print("Steps {} of {}".format(k,val_steps))
                    k1 = k * 10 * batch_size  # Pointer to fetch data from questions and paragraph lists
                    k2 = k * batch_size  # Pointer to fetch data from answers list
                    if k1 + 10 * batch_size <= len(test_ques_list):
                        q_lt = test_ques_list[k1: k1 + 10 * batch_size]
                        p_lt = test_paras_list[k1: k1 + 10 * batch_size]
                    else:
                        q_lt = test_ques_list[len(test_ques_list) - (10 * batch_size): len(test_ques_list)]
                        p_lt = test_paras_list[len(test_ques_list) - (10 * batch_size): len(test_ques_list)]

                    q_v = sess.run(q_vec, feed_dict={q_ph: q_lt})
                    p_v = sess.run(p_vec, feed_dict={p_ph: p_lt})

                    out = sess.run(y_out, feed_dict={ques_ph: q_v, paras_ph: p_v, drop_prob: 0.0})


                    val_reward.append(testing_reward_function(answers=np.array(a_lt), predictions=out))  # np.array, shape: Batch,1,1

                avg_reward = np.mean(val_reward)
                print("Validation Acc = {}".format(avg_reward))

                if avg_reward >= validation_acc:
                    saver.save(sess=sess,
                               save_path="./models/msaic_use_2heads_{}.ckpt".format(j))
                    print("Model msaic_use_2heads_{}.ckpt saved".format(i))
                '''
