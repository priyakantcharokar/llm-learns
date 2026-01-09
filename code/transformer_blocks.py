# transformer_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Self-Attention Head
# ----------------------------
class SelfAttentionHead(nn.Module):
    """
    A single attention head that looks at how words relate to each other.
    
    This computes which words in a sentence should pay attention to which other words.
    It uses three things: keys, queries, and values to figure out the relationships.
    """
    def __init__(self, embedding_dim, block_size, head_size):
        """
        Set up the attention head.
        
        Args:
            embedding_dim: The size of each word's representation
            block_size: The maximum number of words we can look at
            head_size: The size of this attention head
        """
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Process the input and compute attention.
        
        This function:
        1. Creates keys, queries, and values from the input
        2. Calculates how much each word should attend to others
        3. Makes sure words can only look at previous words (not future ones)
        4. Combines everything to get the final output
        
        Args:
            x: Input tensor with shape (batch, sequence_length, embedding_dim)
            
        Returns:
            Output tensor with the same shape as input, but with attention applied
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

# ----------------------------
# Multi-Head Attention
# ----------------------------
class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads working together.
    
    Instead of using just one attention head, we use many heads in parallel.
    Each head can learn to look at different types of relationships between words.
    This helps the model understand the input better.
    """
    def __init__(self, embedding_dim, block_size, num_heads):
        """
        Set up multiple attention heads.
        
        Args:
            embedding_dim: The size of each word's representation
            block_size: The maximum number of words we can look at
            num_heads: How many attention heads to use
        """
        super().__init__()
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dim, block_size, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)

    def forward(self, x):
        """
        Process input through all attention heads and combine the results.
        
        This function:
        1. Runs the input through each attention head separately
        2. Combines all the outputs together
        3. Projects the combined result back to the original size
        
        Args:
            x: Input tensor with shape (batch, sequence_length, embedding_dim)
            
        Returns:
            Output tensor with the same shape as input, combining all attention heads
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

# ----------------------------
# Feed Forward Network
# ----------------------------
class FeedForward(nn.Module):
    """
    A simple neural network that processes each word independently.
    
    After attention figures out relationships between words, this network
    processes each word on its own to add more complexity and learning.
    """
    def __init__(self, n_embd):
        """
        Set up the feed forward network.
        
        Args:
            n_embd: The size of each word's representation (same as embedding_dim)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    def forward(self, x):
        """
        Process the input through the feed forward network.
        
        This function:
        1. Expands each word's representation to 4 times its size
        2. Applies a ReLU activation (makes negative values zero)
        3. Shrinks it back to the original size
        
        Args:
            x: Input tensor with shape (batch, sequence_length, embedding_dim)
            
        Returns:
            Output tensor with the same shape as input, after processing
        """
        return self.net(x)

# ----------------------------
# Transformer Block
# ----------------------------
class Block(nn.Module):
    """
    A complete transformer block that combines attention and feed forward.
    
    This is one layer of the transformer. It does two main things:
    1. Multi-head attention to understand relationships between words
    2. Feed forward network to process each word individually
    
    It uses layer normalization and skip connections to help with training.
    """
    def __init__(self, embedding_dim, block_size, n_heads):
        """
        Set up a transformer block.
        
        Args:
            embedding_dim: The size of each word's representation
            block_size: The maximum number of words we can look at
            n_heads: How many attention heads to use
        """
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Process the input through the transformer block.
        
        This function:
        1. Normalizes the input, applies attention, and adds it back (skip connection)
        2. Normalizes again, applies feed forward, and adds it back (skip connection)
        
        The skip connections (adding x back) help the model learn better by
        allowing information to flow directly through the block.
        
        Args:
            x: Input tensor with shape (batch, sequence_length, embedding_dim)
            
        Returns:
            Output tensor with the same shape as input, after processing through the block
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x