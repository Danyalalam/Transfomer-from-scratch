import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Load dataset and tokenizer modules
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

# Function to extract all sentences from the dataset in a specific language
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# Function to either load a tokenizer if it exists, or build one from the dataset
def get_or_build_tokenizer(config, ds, lang):
    # Define the tokenizer path where it's stored
    tokenizer_path = Path(config['tokenizer_path'], format(lang))
    
    # If tokenizer does not exist, create a new one
    if not Path.exists(tokenizer_path):
        # Define a tokenizer using WordLevel (word-level tokenization)
        tokenizer = Tokenizer(WordLevel(unk_token= '[UNK]'))
        # Use whitespace to tokenize the text
        tokenizer.pre_tokenizers = Whitespace()
        
        # Define a trainer with some special tokens
        trainer = WordLevelTrainer(
            special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'],
            min_frequency=2  # Ignore words that appear less than 2 times
        )
        
        # Train the tokenizer from the dataset's sentences
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        # Save the tokenizer to disk
        tokenizer.save(str(tokenizer_path))
    else:
        # Load an existing tokenizer from file
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Return the tokenizer
    return tokenizer

# Function to load the dataset and tokenize it
def get_ds(config):
    # Load raw dataset from 'opus_books' with source and target languages
    ds_raw = load_dataset('opus_books', f'{config["source_lang"]}-{config["target_lang"]}', split="train")
    
    # Build or load tokenizers for the source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Split dataset into training (90%) and validation (10%) sets
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

