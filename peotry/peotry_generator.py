from peotry.Config import Config
from peotry.DataProcessor import DataProcessor
from peotry.model import LSTMModel

config = Config()
data_processor = DataProcessor()
word2idx_dic, idx2word, words, contents = data_processor.preprocess_data()

lstm_model = LSTMModel(config=config, contents=contents, words=words, idx2word=idx2word, word2idx_dic=word2idx_dic)
if lstm_model.model is None:
    lstm_model.train()

# self.config = config
# self.model = None
# self.contents = contents
# self.words = words
# self.word2idx_dic = word2idx_dic
lstm_model.predict_sen("阴平阴平田,")