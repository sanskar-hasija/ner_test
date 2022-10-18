import argparse
from pretrain import pre_train
from train import train_model
from evaluate import evaluate_model_cr, evaluate_model_metrics
import os
os.environ["WANDB_DISABLED"] = "true"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--input_dir',type=str, default= 'data/data.csv'
    )
    parser.add_argument(
        '--model_name_or_path', type=str, default='allenai/scibert_scivocab_uncased'
    )
    parser.add_argument(
        '--pre_train', type=bool, default=True
    )
    parser.add_argument(
        '--pre_train_block_size', type=int, default=128
    )
    parser.add_argument(
        '--pre_train_epochs', type=int, default=10
    )
    parser.add_argument(
        '--pre_train_batch_size', type=int, default=64
    )
    parser.add_argument(
        '--pre_train_save_steps', type=int, default=1000
    )
    parser.add_argument(
        '--train_epochs', type=int, default=5
    )
    parser.add_argument(
        '--train_batch_size', type=int, default=16
    )
    
    
    args = parser.parse_args()
    
    model_path = args.model_name_or_path
    if args.pre_train:
        pre_train(
            args.input_dir,
            args.pre_train_block_size,
            args.model_name_or_path,
            args.pre_train_epochs,
            args.pre_train_batch_size,
            args.pre_train_save_steps,
        )
        model_path = "models/pretrained/"
        
    preds, labels, preds_list, label_list = train_model(
        args.input_dir,  
        args.train_epochs, 
        args.train_batch_size, 
        model_path
    )
    cr = evaluate_model_cr(preds, labels)
    print(cr)
    met = evaluate_model_metrics(preds_list, label_list)
    print(met)
    
    
    
        
