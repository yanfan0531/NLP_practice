# -*- coding: utf-8 -*-
# @Time    : 2019/5/14 6:11 PM
# @Author  : yan

import numpy as np
import tensorflow as tf

# sample data provided
sentences = ["i like dog", "i love coffee", "i hate milk"]

# build vocabulary
tokenizer = tf.keras.preprocessing.text.Tokenizer()  # num_words=20, oov_token='unk' are optional
tokenizer.fit_on_texts(sentences)   # build vocabulary
train_idxs = tokenizer.texts_to_sequences(sentences)    # text to sequence of idx, indexing from 1 instead of 0!!!
print(train_idxs)

# build word2id, id2word
word2id = tokenizer.word_index
word2id['pad'] = 0
id2word = {v: k for k, v in word2id.items()}
# print(word2id)
# print(id2word)


def generate_batches(train_idxs, num_context_grams):
    input = []
    output = []
    for idx_list in train_idxs:
        for i in range(0, len(idx_list)-num_context_grams):
            context_list = idx_list[i:i+num_context_grams]
            target_word = idx_list[i+num_context_grams]
            # simple features as one-hot vector
            input.append(np.eye(len(word2id))[context_list, :])
            output.append(np.eye(len(word2id))[target_word, :])
    return input, output

# generate batches for model
num_context_grams = 2
input_batch, output_batch = generate_batches(train_idxs, num_context_grams)
# print(input_batch)
# print(output_batch)

# define input and output
vocab_size = len(word2id)
X = tf.placeholder(tf.float32, [None, num_context_grams, vocab_size])  #[batch_size, num_grams, vocab_size]
Y = tf.placeholder(tf.float32, [None, vocab_size])

# define the graph
n_hidden1 = 2
input = tf.reshape(X, [-1, num_context_grams*vocab_size])

W1 = tf.Variable(tf.random_normal([num_context_grams*vocab_size, n_hidden1]), name='W1')
b1 = tf.Variable(tf.random_normal([n_hidden1]), name='b1')
W2 = tf.Variable(tf.random_normal([n_hidden1, vocab_size]), name='W2')
b2 = tf.Variable(tf.random_normal([vocab_size]), name='b2')

output1 = tf.nn.tanh(tf.matmul(input, W1) + b1)
logits_y = tf.matmul(output1, W2) + b2  # not for softmax here
predict_y = tf.argmax(logits_y, -1)

# define loss function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_y, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# lauch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(5000):
        _, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y:output_batch})
        if (epoch+1) % 500 == 0:
            print('Epoch:', '%4d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    # print sample result
    prediction = sess.run(predict_y, feed_dict={X:input_batch, Y:output_batch})
    for idx, idx_list in enumerate(train_idxs):
        for i in range(0, len(idx_list)-num_context_grams):
            context_list = [id2word[id] for id in idx_list[i:i+num_context_grams]]
            target_word = id2word[idx_list[i+num_context_grams]]
            print('context:', context_list)
            print('true target:', target_word)
            print('predict target:', id2word[prediction[idx]], '\n')

