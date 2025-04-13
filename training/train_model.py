import os
import random
import numpy as np
import time

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    PreTrainedTokenizerFast
)
from datasets import Dataset
from collections import defaultdict

base_dir = os.path.join(os.path.dirname(__file__), '..')
output_dir = os.path.join(base_dir, 'minihex', 'ONLY_BLUE')
logs_dir = os.path.join(base_dir, 'logs')
tokenizer_dir = os.path.join(base_dir, 'tokenizer_2')
dataset_path = os.path.join(base_dir, 'hexhex_blue.txt')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=260,
    n_ctx=260,
    n_embd=128,
    n_layer=4,
    n_head=4,
    pad_token_id=tokenizer.pad_token_id,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    embd_pdrop=0.1
)
model = GPT2LMHeadModel(config)

def load_data(file_path):
    """Loads lines from a text file into a Huggingface Dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return Dataset.from_dict({'text': lines})

def tokenize_function(examples):
    """
    Tokenizes each example, adding <|startoftext|> and <|endoftext|>.
    The "text" column is preserved for later filtering.
    """
    text_with_special_tokens = ['<|startoftext|>' + text.strip() + '<|endoftext|>' for text in examples['text']]
    return tokenizer(
        text_with_special_tokens,
        padding='max_length',
        truncation=True,
        max_length=260
    )

dataset = load_data(dataset_path)

# 5% for validation
train_size = int(0.95 * len(dataset))
train_dataset_hf = dataset.select(range(train_size))
val_dataset_hf   = dataset.select(range(train_size, len(dataset)))

# Tokenize (keep "text" column for sorting)
tokenized_train_dataset = train_dataset_hf.map(tokenize_function, batched=True)
tokenized_val_dataset   = val_dataset_hf.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

class ResetTrainDataloaderCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer._train_dataloader = None
        return control

class GroupShuffleTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
            
        original_list = list(self.train_dataset)
        if "text" not in original_list[0]:
            raise ValueError("Field 'text' is missing in dataset — sorting not possible.")
        
        rng = random.Random()
        rng.seed(time.time_ns())
        
        length_groups = defaultdict(list)
        for item in original_list:
            word_count = len(item["text"].strip().split())
            length_groups[word_count].append(item)
        
        new_list = []
        for length in sorted(length_groups.keys(), reverse=True):
            group = length_groups[length]
            rng.shuffle(group)
            new_list.extend(group)
        
        new_dataset = Dataset.from_dict({k: [d[k] for d in new_list] for k in new_list[0].keys()})
        
        print("\n=== FIRST 2 lines of the training dataset ===")
        print(new_dataset[0]["text"].strip())
        print(new_dataset[1]["text"].strip())
        print("=== LAST 2 lines of the training dataset ===")
        print(new_dataset[-2]["text"].strip())
        print(new_dataset[-1]["text"].strip())
        
        # Remove the "text" column so the DataCollator can process the inputs correctly
        new_dataset = new_dataset.remove_columns(["text"])
        
        from torch.utils.data import DataLoader, SequentialSampler
        dataloader = DataLoader(
            new_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=SequentialSampler(new_dataset),
            collate_fn=self.data_collator
        )
        return dataloader

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=20,  # Total epochs (will be overridden per loop)
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir=logs_dir,
    logging_steps=100,
    save_strategy="no",       # Disable automatic saving
    evaluation_strategy="no", # Disable automatic evaluation
    save_total_limit=100,
    eval_accumulation_steps=10,
    save_steps=10000,
    learning_rate=1e-4,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)


# Get unique line lengths (by word count) from the original training dataset
unique_lengths = sorted(list({len(item["text"].strip().split()) for item in train_dataset_hf}), reverse=True)
print("Unique line lengths (descending):", unique_lengths)

# Iterative training on groups of lines
for length in unique_lengths:
    # Filter tokenized dataset to include only lines with the given length
    group_dataset = tokenized_train_dataset.filter(lambda x: len(x["text"].strip().split()) == length)
    print(f"\nTraining on lines of length {length}. Number of examples: {len(group_dataset)}")
    
    trainer = GroupShuffleTrainer(
        model=model,
        args=training_args,
        train_dataset=group_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
    )
    trainer.add_callback(ResetTrainDataloaderCallback())
    
    trainer.args.num_train_epochs = 10
    
    trainer.train()
    
    eval_results = trainer.evaluate(eval_dataset=trainer.eval_dataset)
    print(f"Evaluation loss after training on length {length}: {eval_results['eval_loss']}")
    
    save_path = os.path.join(output_dir, f"checkpoint_length_{length}")
    trainer.save_model(save_path)
    
    model = GPT2LMHeadModel.from_pretrained(save_path)

print("Training complete!")
