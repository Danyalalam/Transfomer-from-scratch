import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # You missed adding parentheses to create a Dropout layer
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Create a matrix of shape (seq_len, d_model) filled with zeros
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of positions from 0 to seq_len-1, reshaped to (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Calculate the division term for even indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sin to the even indices of each position
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to the odd indices of each position
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension to the positional encoding matrix (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        # Register pe as a buffer so it is not treated as a trainable parameter
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.shape[1], :]
        # Apply dropout regularization
        x = self.dropout(x)
        return x
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        # Epsilon value (small constant) to avoid division by zero
        self.eps = eps
        # Learnable scaling factor (alpha), initialized to 1
        self.alpha = nn.Parameter(torch.ones(1))
        # Learnable bias term (beta), initialized to 0
        self.bias = nn.Parameter(torch.zeros(1))  
        
    def forward(self, x):
        # Calculate the mean of the input tensor along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate the standard deviation of the input tensor along the last dimension
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize the input by subtracting the mean and dividing by the standard deviation
        x_normalized = (x - mean) / (std + self.eps)
        
        # Scale the normalized input with learnable alpha and add the bias (learnable beta)
        x_normalized = self.alpha * x_normalized + self.bias
        
        return x_normalized
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Fully connected layer from d_model to d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Fully connected layer from d_ff to d_model
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # Apply the first linear transformation
        x = self.fc1(x)
        # Apply the ReLU activation function
        x = self.relu(x)
        # Apply dropout regularization
        x = self.dropout(x)
        # Apply the second linear transformation
        x = self.fc2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        
        # Store the model dimensions and number of heads
        self.d_model = d_model
        self.h = h
        
        # Ensure the model dimension is divisible by the number of heads
        self.d_k = d_model // h  # Dimension of each head

        # Linear layers for Query, Key, and Value projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value

        # Output linear layer (after concatenating heads)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # Dimension of key/query (d_k)
        
        # Calculate attention scores (Q @ K^T) and scale by sqrt(d_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (if provided) to ignore certain positions
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax to convert scores to probabilities
        attention_scores = attention_scores.softmax(dim=-1)
        
        # Apply dropout to attention scores
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Return attention-weighted values and the attention scores
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Project input sequences to query, key, and value spaces
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape into multi-head format: (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention using the query, key, and value
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Reshape back to original dimensions: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Apply the final linear projection (Wo)
        return self.w_o(x)


##Residual connections: These help the model retain the original input while adding the new learned features.
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        # Layer normalization to stabilize the training
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # Normalize the input, apply the sublayer (e.g., attention or feed-forward), 
        # then apply dropout and add the original input (residual connection)
        ##Preserves information: Adding the original input (x) back to the sublayer's output helps prevent the model from losing important information.
        ##Improves gradient flow: Residual connections help gradients propagate through the network more easily during backpropagation, making training more stable.
        return x + self.dropout(sublayer(self.norm(x)))


# Encoder block: Consists of self-attention, feed-forward, and residual connections
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.attention = self_attention_block  # Self-attention block (multi-head attention)
        self.feed_forward = feed_forward_block  # Feed-forward block (fully connected network)
        
        # Two residual connections: one for attention, one for feed-forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # First apply self-attention with residual connection
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, src_mask))
        
        # Then apply the feed-forward network with residual connection
        x = self.residual_connection[1](x, lambda x: self.feed_forward)
        
        return x


# Encoder: Consists of a stack of encoder blocks and applies layer normalization at the end
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers  # List of encoder blocks
        self.norm = nn.LayerNorm()  # Apply layer normalization at the end

    def forward(self, x, mask):
        ## mask (to prevent attending to certain positions, such as padding tokens).
        # Pass through each encoder block (layer)
        for layer in self.layers:
            x = layer(x, mask)  # Apply each encoder block
        
        # Apply final layer normalization after all layers
        x = self.norm(x)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        # Initialize attention and feed-forward components
        self.self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        
        # Create three residual connections
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Apply self-attention with target mask
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        # Apply cross-attention with source mask
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        # Apply feed-forward network
        x = self.residual_connection[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        # Store the list of decoder layers
        self.layers = layers
        # Create a layer normalization component
        self.norm = nn.LayerNorm()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Apply each decoder layer sequentially
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply layer normalization to the final output
        return self.norm(x)
        
        
class ProjectionLayer(nn.Module):
    def __init__ (self, d_model: int, vocab_size: int):
        super().__init__()
        # Linear layer to project the decoder output to the vocabulary size
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)
    
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        # Initialize encoder and decoder components
        self.encoder = encoder
        self.decoder = decoder
        # Initialize input embedding layers
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        # Initialize positional encoding layers
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        # Initialize projection layer
        self.projection_layer = projection_layer        
        
    def encode(self, src, src_mask):
        # Embed the source input sequences
        x = self.src_pos(self.src_embed(src))
        # Pass the embedded sequences through the encoder
        return self.encoder(x, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, target_mask):
        x = self.src_pos(self.src_embed(tgt))
        # Pass the embedded source and target sequences through the decoder
        return self.decoder(x, encoder_output, src_mask, target_mask)
    
    def project(self, x):
        # Pass the decoder output through the projection layer
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, d_ff: int=2048, dropout: float=0.1) -> Transformer:
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    
    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create encoder layers
    encoder_blocks = []
    encoder_layers = nn.ModuleList([EncoderBlock(MultiHeadAttentionBlock(d_model, h, dropout), FeedForward(d_model, d_ff, dropout), dropout) for _ in range(N)])
    encoder = Encoder(encoder_layers)
    encoder_blocks.append(encoder)
    
    # Create decoder layers
    decoder_blocks = []
    decoder_layers = nn.ModuleList([DecoderBlock(MultiHeadAttentionBlock(d_model, h, dropout), MultiHeadAttentionBlock(d_model, h, dropout), FeedForward(d_model, d_ff, dropout), dropout) for _ in range(N)])
    decoder = Decoder(decoder_layers)
    decoder_blocks.append(decoder)
    
    # Wrap encoder and decoder blocks
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Assemble Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
    