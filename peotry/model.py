import os
import random

import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from keras.engine.saving import load_model
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam


class LSTMModel(object):

    def __init__(self, config, contents, words, idx2word, word2idx_dic, loaded_model=True):
        self.config = config
        self.model = None
        self.contents = contents
        self.words = words
        self.idx2word = idx2word
        self.word2idx_dic = word2idx_dic
        self.words_length = len(words)
        self.loaded_model = loaded_model
        self.poems = contents.split(config.line_delimiter)
        self.poem_num = len(self.poems)
        if os.path.exists(self.config.weight_file) and self.loaded_model:
            self.model = load_model(self.config.weight_file)

    def build_model(self):
        input_tensor = Input(shape=(self.config.max_len, self.words_length))
        lstm = LSTM(units=512, return_sequences=True)(input_tensor)
        dropout = Dropout(0.6)(lstm)
        lstm = LSTM(units=256, )(dropout)
        dropout = Dropout(0.6)(lstm)
        dense = Dense(self.words_length, activation='softmax')(dropout)
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def data_generator(self):
        i = 0
        while 1:
            x = self.contents[i: i + self.config.max_len]
            y = self.contents[i + self.config.max_len]

            if self.config.line_delimiter in x or self.config.line_delimiter in y:
                i += 1
                continue

            y_vec = np.zeros(shape=(1, self.words_length), dtype=np.bool)
            y_vec[0, self.word2idx_dic(y)] = 1.0
            x_vec = np.zeros(shape=(1, self.config.max_len, self.words_length), dtype=np.bool)

            for t, char in enumerate(x):
                x_vec[0, t, self.word2idx_dic(char)] = 1.0

            yield x_vec, y_vec
            i += 1

    def generate_sample_result(self, epoch, logs):
        if epoch % 5 != 0:
            return
        with open(self.config.output_file_path, 'a', encoding='utf-8') as f:
            f.write('==================第{}轮=====================\n'.format(epoch))

        print("\n==================第{}轮=====================".format(epoch))
        for diversity in [0.7, 1.0, 1.3]:
            print("------------设定诗词创作自由度约束参数为{}--------------".format(diversity))
            generate = self.predict_random(temperature=diversity)
            print(generate)
            with open(self.config.output_file_path, 'a', encoding='utf-8') as f:
                f.write(generate + '\n')

    def predict_random(self, temperature=1.):
        if not self.model:
            print("not pre-train model to load.")
            return

        index = random.randint(0, self.poem_num)
        sentence = self.poems[index][: self.config.max_len]
        generate = self.predict_sen(sentence, temperature=temperature)
        return generate

    def predict_sen(self, sentence, temperature=1.):

        if len(sentence) < self.config.max_len:
            print('num of input should greater than ', self.config.max_len)
            return
        sentence = sentence[-self.config.max_len:]
        pre_sent = str(sentence)
        pre_sent += self._preds(sentence, length=24 - self.config.max_len, temperature=temperature)
        return pre_sent

    def _preds(self, sentence, length=23, temperature=1.):
        sentence = sentence[:self.config.max_len]
        sent_pre = ''
        for i in range(length):
            pre = self._pred(sentence=sentence, temperature=temperature)
            sent_pre += pre
            sentence = sentence[1:] + pre
        return sent_pre

    def _pred(self, sentence, temperature=1.):
        sentence = sentence[-self.config.max_len:]
        x_pred = np.zeros((1, self.config.max_len, self.words_length))
        for t, char in enumerate(sentence):
            x_pred[0, t, self.word2idx_dic(char)] = 1.
        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds, temperature=temperature)
        next_char = self.idx2word[next_index]
        return next_char

    def train(self):
        if self.model is None:
            self.build_model()

        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=self.config.epoch_size,
            callbacks=[
                EarlyStopping(),
                ModelCheckpoint(self.config.weight_file, save_weights_only=False, save_best_only=False, verbose=1),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )

    def predict_hide(self, text, temperature=1):
        '''预估模式4：根据给4个字，生成藏头诗五言绝句'''
        if not self.model:
            print('没有预训练模型可用于加载！')
            return
        if len(text) != 4:
            print('藏头诗的输入必须是4个字！')
            return

        index = random.randint(0, self.poem_num)
        # 选取随机一首诗的最后max_len个字+给出的首个文字作为初始输入
        sentence = self.poems[index][1 - self.config.max_len:] + text[0]
        generate = str(text[0])
        print('第一行为 ', sentence)

        for i in range(5):
            next_char = self._pred(sentence, temperature)
            sentence = sentence[1:] + next_char
            generate += next_char

        for i in range(3):
            generate += text[i + 1]
            sentence = sentence[1:] + text[i + 1]
            for i in range(5):
                next_char = self._pred(sentence, temperature)
                sentence = sentence[1:] + next_char
                generate += next_char
        return generate

    def sample(self, preds, temperature=1.):
        '''
        temperature可以控制生成诗的创作自由约束度
        当temperature<1.0时，模型会做一些随机探索，输出相对比较新的内容
        当temperature>1.0时，模型预估方式偏保守
        在训练的过程中可以看到temperature不同，结果也不同
        就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
        '''

        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds, 1 / temperature)
        preds = exp_preds / np.sum(exp_preds)
        pro = np.random.choice(range(len(preds)), 1, p=preds)
        return int(pro.squeeze())
