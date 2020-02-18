from peotry.Config import Config
from peotry.DataLoader import DataLoader


class DataProcessor(Config):
    data_loader = DataLoader()
    poetry_5 = data_loader.load_five_poetry()

    def get_contents(self):
        return self.poetry_5['content'].tolist()

    def get_titles(self):
        return self.poetry_5['title'].tolist()

    def convert_str(self):
        s = ''
        contents = self.get_contents()
        for c in contents:
            s += c
        return s

    def clean_char(self, clean_count=10):
        contents = self.get_contents()
        words_count = {}
        for c in contents:
            words_count = self.cal_words_count(c, words_count)
        clean_key = []
        for c in words_count:
            if words_count[c] < clean_count:
                clean_key.append(c)

        for c in clean_key:
            del words_count[c]
        return words_count

    def to_words(self, data_dict):
        wordPairs = sorted(data_dict.items(), key=lambda x: -x[1])
        words, _ = zip(*wordPairs)
        words += (" ",)
        return words

    def cal_words_count(self, data_str, words_count=None):
        if words_count is None:
            words_count = {}
        words = sorted(data_str)
        for w in words:
            if w in words_count:
                words_count[w] = words_count[w] + 1
            else:
                words_count[w] = 1
        return words_count

    def preprocess_data(self):
        counted_words = self.clean_char()
        words = self.to_words(counted_words)
        word2idx = dict((c, i) for i, c in enumerate(words))
        idx2word = dict((i, c) for i, c in enumerate(words))
        word2idx_dic = lambda x: word2idx.get(x, len(words) - 1)
        return word2idx_dic, idx2word, words, self.convert_str()
