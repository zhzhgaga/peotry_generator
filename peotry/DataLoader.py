import pandas as pd
from peotry.Config import Config


class DataLoader(Config):

    def load_five_poetry(self):
        contents = []
        titles = []
        with open(Config.poetry_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = line.strip() + Config.line_delimiter
                title = d.split(':')[0]
                content = d.split(':')[1]
                if len(content) <= 5:
                    continue
                if content[5] == '，' or content[5] == ',':
                    titles.append(title)
                    contents.append(content)
        return pd.DataFrame(data={'title': titles, 'content': contents})

