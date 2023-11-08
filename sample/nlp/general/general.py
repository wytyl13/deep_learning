'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-10-27 15:34:02
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-10-27 15:34:02
 * @Description: nlp比cv有两个难点，第一如何将文本数字化。第二文本的每个句子都是
 * 不同长度的。不像图像一样我们可以通过预处理来统一化图像的尺寸。
 * 如何解决这个问题？第一使用分词器将句子进行分词处理。然后对分词后的列表进行填充
 * 一般还是零填充，然后我们可以定义填充参数。当然我们可以使用padding参数设置填充位置
 * 还可以设定最大填充长度，并且在此基础上设置阶段位置。
 * 然后统一这些预处理数据以后，我们就可以进行神经网络训练了。就是构建一个可以识别
 * 文本情感的分类器。
 * 我们需要数据集，然后基于它去构建自己的分类器。当然这个数据集是需要标注的
 * 我们会使用到json工具去存储对应的数据
***********************************************************************'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json


if __name__ == "__main__":
    sentences = [
        'I love my dog',
        'I love my cat',
        'I love my funny grandpa',
        'I love my funny wife and like her smile!'
    ]

    # create instancement
    tokenizer = Tokenizer(num_words = 100)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding = 'post', maxlen = 5, truncating = 'post')
    print(word_index)
    print(sequences)
    print(padded)

