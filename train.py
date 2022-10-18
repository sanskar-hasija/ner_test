from simpletransformers.ner import NERModel
from sklearn.model_selection import GroupShuffleSplit 
import pandas as pd
import numpy as np 
import gc
import torch 

def train_model(input_dir,  epochs, batch_size, model_path):
    df = pd.read_csv(input_dir)

    print("Total Number of Unique Sentence: ",len(set(df["sentence_id"].values)))
    df["labels"].fillna("O", inplace = True)


    splitter = GroupShuffleSplit(test_size= 0.2 , n_splits=1, random_state = 12 )
    split = splitter.split(df, groups=df['sentence_id'])
    train_inds, test_inds = next(split)
    train_df = df.iloc[train_inds]
    test_df = df.iloc[test_inds]

    train_df["words"] = train_df["words"].astype("str")
    test_df["words"] = test_df["words"].astype("str")
    train_df.reset_index(drop = True,inplace = True)
    test_df.reset_index(drop = True,inplace = True)

    print("Total Number of Sentences in Train Set: ",len(set(train_df["sentence_id"].values)))
    print("Total Number of Sentences in Test Set: ",len(set(test_df["sentence_id"].values)))

    custom_labels = list(train_df['labels'].unique())
    train_args = {
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'sliding_window': True,
        'max_seq_length': 512,
        'num_train_epochs': epochs,
        'train_batch_size': batch_size,
        'fp16': True,
        'output_dir': 'models/ner/',
        'best_model_dir': 'models/ner/best_model/',
        'evaluate_during_training': True,
    }

    model = NERModel( "bert", model_path, labels=custom_labels, args=train_args)
    model.train_model(train_df, eval_data= test_df)
    result, model_outputs, preds_list = model.eval_model(test_df)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    preds = []
    for p in preds_list:
        preds.extend(p)
    preds = np.array(preds)
    labels =test_df["labels"].values
    
    assert len(preds) == len(labels)

    label_list = []
    for id, p in test_df.groupby("sentence_id"):
        label_list.append(list(p["labels"].values))

    return preds, labels, preds_list, label_list