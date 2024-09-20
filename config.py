from pathlib import Path



def get_config():
    """
    Returns a dictionary containing configuration parameters for the transformer model.
    
    This function defines and returns a set of hyperparameters and settings used 
    throughout the model training and inference process.
    """
    return {
        'source_lang': 'en',  # Source language code (English)
        'target_lang': 'it',  # Target language code (Italian)
        'seq_len': 350,    # Maximum sequence length for input/output
        'batch_size': 8,   # Number of samples per batch during training
        'num_layers': 4,   # Number of encoder and decoder layers in the transformer
        'd_model': 128,    # Dimensionality of the model's internal representations
        'num_heads': 8,    # Number of attention heads in multi-head attention
        'd_ff': 512,       # Dimensionality of the feed-forward networks in transformer layers
        'dropout': 0.1,    # Dropout rate for regularization
        'epochs': 20,      # Number of training epochs
        'lr': 0.0001,      # Learning rate for optimizer
        "model_folder": "weights",  # Directory to save model weights
        "model_basename": "tmodel_",  # Prefix for saved model files
        "preload": None,   # Path to pre-trained model (None if starting from scratch)
        "tokenizer_file": "tokenizer_{0}.json",  # Filename template for tokenizer files
        "experiment_name": "runs/tmodel"  # Name/path for logging experiment results
    }

def get_weights_path(config, epoch: str):
    """
    Generates the full path for saving or loading model weights.

    Args:
        config (dict): The configuration dictionary containing model settings.
        epoch (str): The epoch number or identifier for the weights file.

    Returns:
        str: The full path to the weights file.
    """
    # Extract relevant information from the config
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    
    # Construct the filename for the weights
    model_filename = f"{model_basename}{epoch}.pt"
    
    # Combine the folder path and filename using pathlib for cross-platform compatibility
    return str(Path('.') / model_folder / model_filename)
