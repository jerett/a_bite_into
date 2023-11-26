
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)

        for i in range(max_len):
            for j in range(embed_dim):
                if j % 2 == 0:
                    # pe[0, i, j] = np.sin(i * np.float_power(10000, -j/embed_dim))
                    pe[0, i, j] = np.sin(i * np.power(10000, -j / embed_dim))
                else:
                    # pe[0, i, j] = np.cos(i * np.float_power(10000, -(j-1)/embed_dim))
                    pe[0, i, j] = np.cos(i * np.power(10000, -(j - 1) / embed_dim))
        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        pe = self.pe[:, :S, :]
        output = self.dropout(x + pe)
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below
        # .
        output = torch.empty((N, S, E))
        H = self.n_head

        q = self.query(query)   # (N, S, E)
        k = self.key(key)  # (N, T ,E)
        v = self.value(value)  # (N, T, E)

        # print(q.shape, k.shape, v.shape)
        multi_q = torch.reshape(q, shape=(N, S, H, self.head_dim))  # (N, S, H, E/H)
        multi_q = multi_q.permute(0, 2, 1, 3)  # (N, H, S, E/H)

        multi_k = torch.reshape(k, shape=(N, T, H, self.head_dim))  # (N, T, H, E/H)
        multi_k = multi_k.permute(0, 2, 1, 3)  # (N, H, T, E/H)

        multi_v = torch.reshape(v, shape=(N, T, H, self.head_dim))  # (N, T, H, E/H)
        multi_v = multi_v.permute(0, 2, 1, 3)  # (N, H, T, E/H)

        align = torch.matmul(multi_q, multi_k.permute(0, 1, 3, 2))  # (N, H, S, T)
        align /= np.sqrt(self.head_dim)
        if attn_mask is not None:
            # align = torch.masked_fill(align, (attn_mask==0), np.NINF)
            align = torch.masked_fill(align, (attn_mask==0), -1e9)
        atten = torch.softmax(align, dim=3)
        atten = self.attn_drop(atten)

        multi_y = torch.matmul(atten, multi_v)  # (N, H, S, E/H)
        y = multi_y.permute(0, 2, 1, 3) # (N, S, H, E/H)
        y = y.reshape(shape=(N, S, E))  # (N, S, E)
        y = self.proj(y)  # (N, S, E)
        output = y
        return output


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-attention layer
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        # Feedforward network
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, src_mask=None):
        # src: (N, S, E), src_mask: (S, S)
        # Apply self-attention and add residual
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + src2)
        # Apply feedforward network and add residual
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + src2)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # Stack multiple encoder layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        # src: (N, S, E), src_mask: (S, S)
        # Pass the source through each encoder layer
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-attention layer
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        # Multi-head attention layer
        self.multihead_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        # Feedforward network
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, tgt_mask=None):
         # tgt: (N, T, E), memory: (N, S, E), tgt_mask: (T, T)
        # Self-attention on the target
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + tgt2)
        # Attention between target and encoder output
        tgt2 = self.multihead_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + tgt2)
        # Feedforward network and residual
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + tgt2)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        # Stack multiple decoder layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None):
        # tgt: (N, T, E), memory: (N, S, E), tgt_mask: (T, T)
        # Pass the target through each decoder layer
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask)
        return tgt


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 embed_dim, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, max_seq_length, dropout=0.1):
        super().__init__()
        self.src_embedding = Embeddings(embed_dim, src_vocab_size)
        self.tgt_embedding = Embeddings(embed_dim, tgt_vocab_size)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_seq_length)
        encoder_layer = TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
    def encode(self, src, src_mask=None):
        src = self.src_embedding(src)  # Apply source embedding
        src = self.pos_encoder(src)    # Apply positional encoding
        return self.encoder(src, src_mask)  # Run through the encoder

    def decode(self, memory, tgt, tgt_mask=None):
        tgt = self.tgt_embedding(tgt)  # Apply target embedding
        tgt = self.pos_encoder(tgt)    # Apply positional encoding
        return self.decoder(tgt, memory, tgt_mask)  # Run through the decoder

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(memory, tgt, tgt_mask)
        return output


if __name__ == '__main__':
    # Set parameters for the model and test
    vocab_size = 1000  # Assume vocabulary size is 1000
    embed_dim = 512  # Embedding dimension
    nhead = 8  # Number of heads in multi-head attention
    num_encoder_layers = 6  # Number of encoder layers
    num_decoder_layers = 6  # Number of decoder layers
    dim_feedforward = 2048  # Dimension of feedforward network
    max_seq_length = 50  # Maximum sequence length
    dropout = 0.1  # Dropout rate

    # Create a Transformer model instance
    model = TransformerModel(vocab_size, vocab_size, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout)

    # Fake input data
    batch_size = 32  # Batch size
    src_seq_length = 40  # Source sequence length
    tgt_seq_length = 30  # Target sequence length

    # Generate random source and target sequences
    src = torch.randint(0, vocab_size, (batch_size, src_seq_length))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_length))

    # Forward pass through the model
    output = model(src, tgt)

    # Print output shape
    print("Output shape:", output.shape)  # Expected shape: (batch_size, tgt_seq_length, embed_dim)
    