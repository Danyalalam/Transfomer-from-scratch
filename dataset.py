import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BillingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        """
        Initialize the BilingualDataset.
        
        :param ds: Source dataset
        :param tokenizer_src: Tokenizer for source language
        :param tokenizer_tgt: Tokenizer for target language
        :param lang_src: Source language code
        :param lang_tgt: Target language code
        :param seq_len: Maximum sequence length
        """
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len
        
        # Create tensors for special tokens
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)  # Start of sentence token
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)  # End of sentence token
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)  # Padding token

    def __len__(self):
        """Return the number of items in the dataset"""
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        :param idx: Index of the item
        :return: Dictionary containing processed data for the transformer model
        """
        # Get the source-target pair
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.lang_src]
        tgt_text = src_target_pair['translation'][self.lang_tgt]
        
        # Tokenize the source and target text
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # Calculate the number of padding tokens needed
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # -1 for SOS (EOS will be in label)
       
        # Check if sequence length is exceeded
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sequence length exceeded')
        
        # Create encoder input: [SOS] + tokens + [EOS] + padding
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0
        )
        
        # Create decoder input: [SOS] + tokens + padding
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0
        )
        
        # Create label: tokens + [EOS] + padding
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0
        )
        
        # Assert that all sequences have the correct length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        # Return a dictionary with all the processed data
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # Mask for encoder (1 for non-pad tokens)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),  # Mask for decoder (combines padding and causal mask)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,          
        }
        
def casual_mask(size):
    """
    Create a causal mask for the decoder.
    
    :param size: Size of the square mask
    :return: A boolean mask where True values indicate positions that can be attended to
    """
    # Create an upper triangular matrix
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    # Invert the mask (1 becomes 0 and 0 becomes 1)
    return mask == 0