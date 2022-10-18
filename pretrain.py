import gc
import torch 
import os
from data_processing import pre_train_datagen
from transformers import LineByLineTextDataset
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def pre_train(input_dir, block_size, model_name, epochs, batch_size, save_steps):
    os.makedirs("models/pretrained/", exist_ok = True)
    pre_train_datagen(input_dir)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset= LineByLineTextDataset(tokenizer = tokenizer,
                                   file_path = 'data/pre_train_text.txt',
                                   block_size = block_size )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir='models/pretrained/',
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )
    trainer.train()
    trainer.save_model('models/pretrained/')
    tokenizer.save_pretrained("models/pretrained/", legacy_format=True)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
