import tensorflow as tf
import numpy as np
import json

# Read data from npy
print("Loding Data..")
Questions = np.load("Questions.npy") # 524188, 768
Answers = np.load("Answers.npy") # 524188, 10
Epochs = 3
# Define tf variables
baseline = tf.Variable(0.0,dtype=tf.float32,name="Baseline", trainable=False)
adv = tf.Variable(0.0,dtype=tf.float32,name="Advantage",trainable=False)

# Define reward functions
def reward_function(answers, predictions):
    """
    Calculates MRR (Mean Reciprocal Rank) for predictions
    :param answers: shape Batch,10
    :param predictions: shape Batch,10,1
    :return: shape Batch,1,1
    """
    rewards=[]
    args = np.argmax(answers,axis=1)
    y_pred=predictions
    for i in range(0,len(y_pred)):
        lt1 = y_pred[i][args[i]] >= y_pred[i]
        s = np.sum(lt1)
        if (s < 4.0):
            rewards.append(1.0)
        elif(s < 6.0):
            rewards.append(0.0)
        else:
            rewards.append(0.0)
        #print(s)
        #rewards.append(1.0/s)
    rew = np.array(rewards)
    return rew.reshape(-1,1,1)

# Define Question Augmentation
def question_aug(question_arr):
    return np.reshape(np.tile(question_arr,10),(-1,10,768))

# Define Placeholders
ques_ph = tf.placeholder(dtype=tf.float32,shape=[None,10,768],name="Questions")
para_ph = tf.placeholder(dtype=tf.float32,shape=[None,10,768],name="Paragraphs")
#out_ph = tf.placeholder(dtype=tf.int16,shape=[None,10,1],name="Answers")
drop_prob = tf.placeholder_with_default(0.0,shape=None,name='Dropout_prob') #Default No dropout

# Define Multihead Attention between Questions and Paragraphs

#Head1
o1 = tf.layers.dense(tf.concat([ques_ph,para_ph],axis=2),256,name="head1_o1",activation='relu')
o1_dp = tf.layers.dropout(inputs=o1,rate=drop_prob,name="head1_drop_out")
out1 = tf.layers.dense(o1_dp,1,name="head1_out",activation=None)

#Head2
o2 = tf.layers.dense(tf.concat([ques_ph,para_ph],axis=2),256,name="head2_o1",activation='relu')
o2_dp = tf.layers.dropout(inputs=o2,rate=drop_prob,name="head2_drop_out")
out2 = tf.layers.dense(o2_dp,1,name="head2_out",activation=None)

# Output
y_out = tf.nn.softmax(out1+out2,axis=1)
#y_out = tf.sigmoid(out1 + out2)
#Define Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

init = tf.global_variables_initializer()

# Checkpoint Saver
saver = tf.train.Saver(max_to_keep=10)

# trainings logs
loss_log = []
baseline_log = []
validation_score = 0
validation_score_lt = []
print("Training Started..")
# Start training
with tf.Session() as sess:
    sess.run(init) # initialize global variables
    for i in range(Epochs+1):
        for j in range(0,100): # Paragraphs from 100 to 104 are used for validation
            Paras = np.load("Paragraph_{}.npy".format(j)) # 5000,10,768
            j1 = j * 5000 # Pointer to the Questions and Answers
            Ques= Questions[j1:j1+5000] # 5000,768
            Ans= Answers[j1:j1+5000] # 5000, 10
            Ques1 = question_aug(Ques) # 5000,10,768
            print("Question Shape = {}".format(Ques1.shape))
            del(Ques)
            op = sess.run(y_out,feed_dict={ques_ph:Ques1, para_ph: Paras, drop_prob:0.2})
            reward_t = reward_function(answers=Answers,predictions=op) # np.array Batch,1,1
            adv = reward_t - baseline # Advantage = Reward - baseline # tf.Tensor Batch,1,1
            baseline = 0.9 * baseline + 0.1 * np.mean(reward_t) # Update the baseline based on the mean reward of batch
            loss = tf.reduce_mean(adv * tf.log(y_out))
            train_op = optimizer.minimize(-loss) # Gradient Descent
            training_loss, base,_ = sess.run([loss,baseline,train_op],feed_dict={ques_ph:Ques1, para_ph: Paras, drop_prob:0.2})
            loss_log.append(training_loss)
            baseline_log.append(base)

            # Print Log
            print("Epoch: {}, Paragraph No:{}".format(i,j))
            print("Loss:{}, Baseline:{}".format(training_loss,base))


            # Run validation after 20K Ques, Paragraphs
            if(j % 4 ==0):
                print("Validation Begin...\n")
                rew = []
                for k in range(100,104):
                    p = np.load("Paragraph_{}.npy".format(k))
                    k1 = k * 5000
                    if(k1 + 5000 > len(Questions)):
                        q = Questions[k1:len(Questions)]
                        a = Answers[k1:len(Answers)]
                    else:
                        q = Questions[k1:k1+5000]
                        a = Answers[k1:k1+5000]
                    val_out = sess.run(y_out,feed_dict={ques_ph:question_aug(q), para_ph:p, drop_prob:0.0})
                    r = reward_function(answers=a,predictions=val_out)
                    rew.append(r)
                print("Validation Finished..\n")
                print("Present Validation Score = {}".format(validation_score))
                print("New Validation Score={}".format(np.mean(rew)))
                validation_score_lt.append(np.mean(rew))
                if(np.mean(rew) >= validation_score):
                    print("New Validation score is high\nSaving model..")
                    validation_score = np.mean(rew)
                    saver.save(sess=sess,
                               save_path="./models/msaic_2head.ckpt")
                    print("Model Saved")

# Save logs in json file for visualization
logs = {}
logs['loss'] = loss_log
logs['baseline'] = baseline_log
logs['validation'] = validation_score_lt


with open('logs.json', 'w') as outfile:
    json.dump(logs, outfile)

