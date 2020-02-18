from os import path


class Config:
    data_path = '/data/pro/poetry'
    poetry_file_path = path.join(data_path, 'poetry.txt')
    output_file_path = path.join(data_path,'out','out.txt')

    weight_file = path.join(data_path, 'model', 'poetry_model.h5')
    max_len = 6
    batch_size = 32
    epoch_size = 10
    learning_rate = 0.003
    line_delimiter = '|||'
