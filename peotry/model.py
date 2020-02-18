import random

import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam


class LSTMModel(object):

    def __init__(self, config, contents, words, idx2word,word2idx_dic):
        self.config = config
        self.model = None
        self.contents = contents
        self.words = words
        self.idx2word = idx2word
        self.word2idx_dic = word2idx_dic
        self.words_length = len(words)
        self.poems = contents.split(config.line_delimiter)
        self.poem_num = len(self.poems)

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

            # 训练时的预测结果写入txt
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
                ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )

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
