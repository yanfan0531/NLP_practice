# -*- coding: utf-8 -*-
# @Time    : 2019/5/16 4:36 PM
# @Author  : yan

"""
demo for the usage of tf.keras.preprocessing.text.Tokenizer
"""
import tensorflow as tf

# sample data (list of documents, each doc is a str where grams are split by space)
sentences = ['i like dog do you', 'i love coffee with milk', 'i hate milk']

# generate an instance of tokenizer
# params: num_words=20000, filter='!"#$%&()*+-/,.:;<=>?@[\\]^_`{|}~\t\n',' oov_token='unk' would be index 1
tokenizer = tf.keras.preprocessing.text.Tokenizer()
print(tokenizer.word_index)

# build vocabulary
# IMPORTANT!!! index 0 reserved for further use, e.g. padding
tokenizer.fit_on_texts(sentences)

# generate sequence of indices, same dimension as sample data (no sign of 0)
train_idxs = tokenizer.texts_to_sequences(sentences)
print(train_idxs)

# generate matrix of indices, dim is (len of doc, num of words) (count for each word)
train_matrix = tokenizer.texts_to_matrix(sentences)
print(train_matrix)
