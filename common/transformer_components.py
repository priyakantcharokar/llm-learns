# File containing transformer architecture components
# transformer_components.py

# PyTorch library for tensor operations
import torch
# Neural network modules and layers
import torch.nn as nn
# Functional interface for neural network operations
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
        # Initialize parent nn.Module class
        super().__init__()
        # Linear layer to compute key vectors
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        # Linear layer to compute query vectors
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        # Linear layer to compute value vectors
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        # Lower triangular mask for causal attention
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def compute_keys_queries_values(self, x):
        """
        Transform input into keys, queries, and values for attention computation.
        
        Args:
            x: Input tensor with shape (batch, sequence_length, embedding_dim)
            
        Returns:
            keys: Key representations
            queries: Query representations
            values: Value representations
        """
        # Transform input to key representations
        keys = self.key(x)
        # Transform input to query representations
        queries = self.query(x)
        # Transform input to value representations
        values = self.value(x)
        # Return all three representations
        return keys, queries, values
    
    def compute_attention_scores(self, queries, keys, sequence_length):
        """
        Calculate attention scores between queries and keys.
        
        Args:
            queries: Query tensor
            keys: Key tensor
            sequence_length: Length of the current sequence
            
        Returns:
            Attention scores (before masking and softmax)
        """
        # Get embedding dimension from query tensor
        embedding_dim = queries.shape[-1]
        # Compute dot product attention scores, scaled by sqrt of embedding dimension
        # Scaled dot-product attention
        attention_scores = queries @ keys.transpose(-2, -1) / (embedding_dim ** 0.5)
        # Return raw attention scores
        return attention_scores
    
    def apply_causal_mask(self, attention_scores, sequence_length):
        """
        Apply causal masking to prevent attending to future tokens.
        
        Args:
            attention_scores: Raw attention scores
            sequence_length: Length of the current sequence
            
        Returns:
            Masked attention scores (future positions set to -inf)
        """
        # Create mask: only allow attention to previous tokens
        # Extract lower triangular mask for current sequence
        causal_mask = self.tril[:sequence_length, :sequence_length]
        # Set future positions to negative infinity (will become 0 after softmax)
        # Mask out future tokens
        masked_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        # Return masked attention scores
        return masked_scores
    
    def apply_attention_weights(self, attention_weights, values):
        """
        Apply attention weights to values to get the final output.
        
        Args:
            attention_weights: Softmax-normalized attention weights
            values: Value representations
            
        Returns:
            Weighted sum of values based on attention weights
        """
        # Weighted sum of values using attention weights
        output = attention_weights @ values
        # Return final attention output
        return output
    
    def forward(self, x):
        """
        Process the input and compute self-attention.
        
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
        # Extract dimensions from input tensor
        batch_size, sequence_length, embedding_dim = x.shape
        # Compute keys, queries, and values
        # Transform input to K, Q, V
        keys, queries, values = self.compute_keys_queries_values(x)
        # Calculate attention scores
        # Compute attention scores
        attention_scores = self.compute_attention_scores(queries, keys, sequence_length)
        # Apply causal masking (prevent seeing future tokens)
        # Mask future positions
        masked_scores = self.apply_causal_mask(attention_scores, sequence_length)
        # Convert scores to probabilities using softmax
        # Normalize to probability distribution
        attention_weights = F.softmax(masked_scores, dim=-1)
        # Apply attention weights to values
        # Weighted combination of values
        output = self.apply_attention_weights(attention_weights, values)
        # Return attention output
        return output

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
        # Initialize parent nn.Module class
        super().__init__()
        # Calculate size of each attention head
        head_size = embedding_dim // num_heads
        # Create multiple attention heads
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dim, block_size, head_size) for _ in range(num_heads)])
        # Projection layer to combine head outputs
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)

    def apply_all_attention_heads(self, x):
        """
        Apply all attention heads in parallel to the input.
        
        Args:
            x: Input tensor with shape (batch, sequence_length, embedding_dim)
            
        Returns:
            List of outputs from each attention head
        """
        # Apply each attention head to input in parallel
        head_outputs = [head(x) for head in self.heads]
        # Return list of outputs from all heads
        return head_outputs
    
    def combine_head_outputs(self, head_outputs):
        """
        Concatenate outputs from all attention heads.
        
        Args:
            head_outputs: List of tensors from each attention head
            
        Returns:
            Concatenated tensor with all head outputs combined
        """
        # Concatenate all head outputs along embedding dimension
        combined = torch.cat(head_outputs, dim=-1)
        # Return concatenated tensor
        return combined
    
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
        # Apply all attention heads in parallel
        # Get outputs from all attention heads
        head_outputs = self.apply_all_attention_heads(x)
        # Concatenate all head outputs
        # Concatenate head outputs
        combined_outputs = self.combine_head_outputs(head_outputs)
        # Project back to original embedding dimension
        # Project to original embedding size
        final_output = self.proj(combined_outputs)
        # Return final multi-head attention output
        return final_output

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
        # Initialize parent nn.Module class
        super().__init__()
        # Create sequential network
        self.net = nn.Sequential(
            # Expand embedding dimension by 4x
            nn.Linear(n_embd, 4 * n_embd),
            # Apply ReLU activation (non-linearity)
            nn.ReLU(),
            # Contract back to original embedding dimension
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
        # Process input through feedforward network
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
        # Initialize parent nn.Module class
        super().__init__()
        # Multi-head self-attention layer
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        # Feedforward neural network layer
        self.ffwd = FeedForward(embedding_dim)
        # Layer normalization before attention
        self.ln1 = nn.LayerNorm(embedding_dim)
        # Layer normalization before feedforward
        self.ln2 = nn.LayerNorm(embedding_dim)

    def apply_attention_with_skip_connection(self, x):
        """
        Apply multi-head attention with layer normalization and skip connection.
        
        Args:
            x: Input tensor with shape (batch, sequence_length, embedding_dim)
            
        Returns:
            Output after attention with residual connection
        """
        # Normalize input, apply attention, add residual connection
        # Normalize input before attention
        normalized_input = self.ln1(x)
        # Apply multi-head attention
        attention_output = self.sa(normalized_input)
        # Add residual connection (skip connection)
        output_with_residual = x + attention_output
        # Return output with residual
        return output_with_residual
    
    def apply_feedforward_with_skip_connection(self, x):
        """
        Apply feedforward network with layer normalization and skip connection.
        
        Args:
            x: Input tensor with shape (batch, sequence_length, embedding_dim)
            
        Returns:
            Output after feedforward with residual connection
        """
        # Normalize input, apply feedforward, add residual connection
        # Normalize input before feedforward
        normalized_input = self.ln2(x)
        # Apply feedforward network
        feedforward_output = self.ffwd(normalized_input)
        # Add residual connection (skip connection)
        output_with_residual = x + feedforward_output
        # Return output with residual
        return output_with_residual
    
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
        # Apply attention with skip connection
        # Process through attention + residual
        x = self.apply_attention_with_skip_connection(x)
        # Apply feedforward with skip connection
        # Process through feedforward + residual
        x = self.apply_feedforward_with_skip_connection(x)
        # Return final transformer block output
        return x

