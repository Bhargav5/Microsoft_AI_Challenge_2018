import tensorflow as tf
import numpy as np

ques_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='questions')
para1_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph1')
para2_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph2')
para3_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph3')
para4_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph4')
para5_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph5')
para6_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph6')
para7_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph7')
para8_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph8')
para9_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph9')
para10_ph = tf.placeholder(dtype=tf.float32,shape=[None,768],name='Paragraph10')
labels_ph = tf.placeholder(dtype=tf.int16,shape=[None,10],name='Labels')



with tf.name_scope("Head1"):
    with tf.variable_scope("head1"):
        o1_1 = tf.layers.dense([ques_ph, para1_ph], 256, activation='relu', name='o1')
        o1_11 = tf.layers.dropout(o1_1, rate=0.75, name='o11')
        out1_1 = tf.layers.dense(o1_11, 1, name='out1')
    with tf.variable_scope("head1", reuse=True):
        o1_2 = tf.layers.dense([ques_ph, para2_ph], 256, activation='relu', name='o1')
        o1_21 = tf.layers.dropout(o1_2, rate=0.75, name='o11')
        out1_2 = tf.layers.dense(o1_21, 1, name='out1')

        o1_3 = tf.layers.dense([ques_ph, para3_ph], 256, activation='relu', name='o1')
        o1_31 = tf.layers.dropout(o1_3, rate=0.75, name='o11')
        out1_3 = tf.layers.dense(o1_31, 1, name='out1')

        o1_4 = tf.layers.dense([ques_ph, para4_ph], 256, activation='relu', name='o1')
        o1_41 = tf.layers.dropout(o1_4, rate=0.75, name='o11')
        out1_4 = tf.layers.dense(o1_41, 1, name='out1')

        o1_5 = tf.layers.dense([ques_ph, para5_ph], 256, activation='relu', name='o1')
        o1_51 = tf.layers.dropout(o1_5, rate=0.75, name='o11')
        out1_5 = tf.layers.dense(o1_51, 1, name='out1')

        o1_6 = tf.layers.dense([ques_ph, para6_ph], 256, activation='relu', name='o1')
        o1_61 = tf.layers.dropout(o1_6, rate=0.75, name='o11')
        out1_6 = tf.layers.dense(o1_61, 1, name='out1')

        o1_7 = tf.layers.dense([ques_ph, para7_ph], 256, activation='relu', name='o1')
        o1_71 = tf.layers.dropout(o1_7, rate=0.75, name='o11')
        out1_7 = tf.layers.dense(o1_71, 1, name='out1')

        o1_8 = tf.layers.dense([ques_ph, para8_ph], 256, activation='relu', name='o1')
        o1_81 = tf.layers.dropout(o1_8, rate=0.75, name='o11')
        out1_8 = tf.layers.dense(o1_81, 1, name='out1')

        o1_9 = tf.layers.dense([ques_ph, para9_ph], 256, activation='relu', name='o1')
        o1_91 = tf.layers.dropout(o1_9, rate=0.75, name='o11')
        out1_9 = tf.layers.dense(o1_91, 1, name='out1')

        o1_10 = tf.layers.dense([ques_ph, para10_ph], 256, activation='relu', name='o1')
        o1_101 = tf.layers.dropout(o1_10, rate=0.75, name='o11')
        out1_10 = tf.layers.dense(o1_101, 1, name='out1')
# Second Head
with tf.name_scope("Head2"):
    with tf.variable_scope("head2"):
        o2_1 = tf.layers.dense([ques_ph, para1_ph], 256, activation='relu', name='o2')
        o2_11 = tf.layers.dropout(o2_1, rate=0.75, name='o21')
        out2_1 = tf.layers.dense(o2_11, 1, name='out2')
    with tf.variable_scope("head2", reuse=True):
        o2_2 = tf.layers.dense([ques_ph, para2_ph], 256, activation='relu', name='o2')
        o2_21 = tf.layers.dropout(o2_2, rate=0.75, name='o21')
        out2_2 = tf.layers.dense(o2_21, 1, name='out2')

        o2_3 = tf.layers.dense([ques_ph, para3_ph], 256, activation='relu', name='o2')
        o2_31 = tf.layers.dropout(o2_3, rate=0.75, name='o21')
        out2_3 = tf.layers.dense(o2_31, 1, name='out2')

        o2_4 = tf.layers.dense([ques_ph, para4_ph], 256, activation='relu', name='o2')
        o2_41 = tf.layers.dropout(o2_4, rate=0.75, name='o21')
        out2_4 = tf.layers.dense(o2_41, 1, name='out2')

        o2_5 = tf.layers.dense([ques_ph, para5_ph], 256, activation='relu', name='o2')
        o2_51 = tf.layers.dropout(o2_5, rate=0.75, name='o21')
        out2_5 = tf.layers.dense(o2_51, 1, name='out2')

        o2_6 = tf.layers.dense([ques_ph, para6_ph], 256, activation='relu', name='o2')
        o2_61 = tf.layers.dropout(o2_6, rate=0.75, name='o21')
        out2_6 = tf.layers.dense(o2_61, 1, name='out2')

        o2_7 = tf.layers.dense([ques_ph, para7_ph], 256, activation='relu', name='o2')
        o2_71 = tf.layers.dropout(o2_7, rate=0.75, name='o21')
        out2_7 = tf.layers.dense(o2_71, 1, name='out2')

        o2_8 = tf.layers.dense([ques_ph, para8_ph], 256, activation='relu', name='o2')
        o2_81 = tf.layers.dropout(o2_8, rate=0.75, name='o21')
        out2_8 = tf.layers.dense(o2_81, 1, name='out2')

        o2_9 = tf.layers.dense([ques_ph, para9_ph], 256, activation='relu', name='o2')
        o2_91 = tf.layers.dropout(o2_9, rate=0.75, name='o21')
        out2_9 = tf.layers.dense(o2_91, 1, name='out2')

        o2_10 = tf.layers.dense([ques_ph, para10_ph], 256, activation='relu', name='o2')
        o2_101 = tf.layers.dropout(o2_10, rate=0.75, name='o21')
        out2_10 = tf.layers.dense(o2_101, 1, name='out2')