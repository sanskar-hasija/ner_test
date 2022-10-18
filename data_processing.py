import pandas as pd
import os

def pre_train_datagen(data_dir):
    data = pd.read_csv(data_dir)
    data = data.dropna(subset = ["words"])
    os.makedirs("data", exist_ok = True)
    with open('data/pre_train_text.txt','w') as f:
        for idx, df in data.groupby("sentence_id"):
            sentence = " ".join(df["words"])
            f.write(sentence+'\n')
    