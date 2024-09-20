import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from dataset import BillingualDataset, casual_mask
from model import build_transformer

from config import get_config, get_weights_path
from tqdm import tqdm
import warnings

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
    
    
    # Create instances of BilingualDataset for training and validation
    # This custom dataset class prepares the raw data for the model
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Initialize variables to track the maximum sequence lengths in the dataset
    max_len_src = 0
    max_len_tgt = 0

    # Iterate through the raw dataset to find the maximum sequence lengths
    for item in ds_raw:
        # Tokenize the source and target sentences
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        
        # Update the maximum lengths if the current sentence is longer
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # Print the maximum sequence lengths found
    # This information can be useful for setting or verifying the 'seq_len' parameter
    print(f'Maximum source length: {max_len_src}')
    print(f'Maximum target length: {max_len_tgt}')

    # Create a DataLoader for the training dataset
    # This wraps the dataset and handles batching and shuffling
    train_data_loader = DataLoader(
        train_ds,                     # The training dataset
        batch_size=config['batch_size'], # Batch size from configuration
        shuffle=True                  # Shuffle data for each epoch
    )

    # Create a DataLoader for the validation dataset
    # Note: batch_size=1 for more granular evaluation, and no shuffling
    val_data_loader = DataLoader(
        val_ds,         # The validation dataset
        batch_size=1,   # Process one sample at a time during validation
        shuffle=False   # No need to shuffle validation data
    )

    # Return the prepared data loaders and tokenizers
    # These will be used in the main training loop and for inference
    return train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt
def get_model(config, vocab_tgt_len, vocab_src_len):
    """
    Creates and returns a transformer model based on the given configuration.

    Args:
        config (dict): Configuration parameters for the model.
        vocab_tgt_len (int): Size of the target language vocabulary.
        vocab_src_len (int): Size of the source language vocabulary.

    Returns:
        nn.Module: The constructed transformer model.
    """
    model = build_transformer(
        num_layers=config['num_layers'],      # Number of encoder and decoder layers
        d_model=config['d_model'],            # Dimension of the model's internal representations
        num_heads=config['num_heads'],        # Number of attention heads
        d_ff=config['d_ff'],                  # Dimension of the feed-forward networks
        input_vocab_size=vocab_src_len,       # Size of the source language vocabulary
        target_vocab_size=vocab_tgt_len,      # Size of the target language vocabulary
        max_pos_len=config['seq_len'],        # Maximum sequence length
        dropout=config['dropout']             # Dropout rate for regularization
    )
    return model

def train_model(config):
    """
    Trains the transformer model using the provided configuration.

    Args:
        config (dict): Configuration parameters for training.
    """
    # Set the device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    
    # Create the model folder if it doesn't exist
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # Get data loaders and tokenizers
    train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Initialize the model and move it to the appropriate device
    model = get_model(config, tokenizer_tgt.get_vocab_size(), tokenizer_src.get_vocab_size()).to(device)
    
    # Initialize TensorBoard writer for logging
    writer = SummaryWriter(config['experiment_name'])
    
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    # Load pre-trained weights if specified
    if config['preload']:
        model_filename = get_weights_path(config, config['preload'])
        print(f'Loading model weights from {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    # Initialize the loss function with label smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    # Training loop
    for epoch in range(initial_epoch, config['epochs']):
        model.train()
        batch_iterator = tqdm(train_data_loader, desc=f'Epoch {epoch:02d}')
        for batch in batch_iterator:
            # Move batch data to the appropriate device
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)   
            label = batch['label'].to(device)
            
            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, decoder_mask, encoder_output, encoder_mask)
            proj_output = model.project(decoder_output)
            
            # Compute loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # Log the loss to TensorBoard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Save the model at the end of every epoch
        model_filename = get_weights_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
            
            
    