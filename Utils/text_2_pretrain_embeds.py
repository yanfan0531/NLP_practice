# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 10:31 AM
# @Author  : yan

"""
this script is used for text preprocessing,
from text to emb_vector fed into tensorflow.
"""
import os
import numpy as np
import tensorflow as tf

RANDOM_SEED = 4
EMBEDDING_DIM = 50
UNK_STR = '<UNK>'
EMBEDDING_PATH = '/Users/yan/Downloads/glove.6B'
EMBEDDING_FILENAME = 'glove.6B.50d.txt'


def load_glove(filepath):
    print('start loading glove embeddings...')
    word_index = dict()
    embed = []

    count = 0
    with open(filepath, 'r') as file:
        for line in file:
            cols = line.strip().split(' ')
            word_index[cols[0]] = count
            embed.append(cols[1:])
            count += 1

    embed = np.array(embed)

    # finally add unk embedding at the bottom end
    word_index[UNK_STR] = count
    np.random.seed(RANDOM_SEED)
    embed = np.concatenate((embed, np.random.randn(1, embed.shape[-1])), axis=0)

    print('glove loading done, with %d vocab (including unk)' % len(word_index))
    return word_index, embed


if __name__ == '__main__':

    # load Glove pretrained embeddings
    word_index, embed = load_glove(os.path.join(EMBEDDING_PATH, EMBEDDING_FILENAME))

    # sample data to list of word index
    sentences = 'the roof is red .'
    train_idx = []
    for s in sentences.split(' '):
        if s in word_index.keys():
            train_idx.append(word_index[s])
        else:
            train_idx.append(word_index[UNK_STR])
    print(train_idx)

    # build Emebedding layer
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
    embed_matrix = tf.Variable(embed, dtype=np.float32)
    input_emb = tf.nn.embedding_lookup(embed_matrix, input_ids)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        emb = sess.run(input_emb, feed_dict={input_ids:train_idx})
        print(emb)
