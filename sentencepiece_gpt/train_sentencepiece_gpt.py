# PyTorch library for tensor operations and neural networks
import torch
# Neural network modules (layers, activations, etc.)
import torch.nn as nn
# Functional interface for neural network operations
import torch.nn.functional as F
# For random number generation
import random
# SentencePiece library for subword tokenization
import sentencepiece as spm


# Add parent directory to path to access common folder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import transformer block components
from common.transformer_components import Block


# Display PyTorch version
print("Torch version:", torch.__version__)
# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())
# Display GPU name if available
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")



# Open the training data file
with open("data.txt", "r", encoding="utf-8") as f:
    # Read all text content from the file
    text = f.read()

# Train a SentencePiece tokenizer model
spm.SentencePieceTrainer.Train(
    # Input file to train on
    input="data.txt",
    # Prefix for output model files
    model_prefix="tokenizer",
    # Size of the vocabulary (number of subword tokens)
    vocab_size=40,
    # Use Byte Pair Encoding algorithm
    model_type="bpe"
)

# Create a SentencePiece processor instance
sp = spm.SentencePieceProcessor()
# Load the trained tokenizer model
sp.load("tokenizer.model")
    
# Convert text to token IDs (list of integers)
ids = sp.encode(text, out_type=int)
# Convert token IDs to PyTorch tensor
data = torch.tensor(ids, dtype=torch.long)

# Display the tokenized data
print(data)

# Get the vocabulary size from the tokenizer
vocab_size = sp.get_piece_size()
# Display the vocabulary size
print(vocab_size)


# Maximum context length (number of tokens the model can see)
block_size = 6
# Dimension of token embeddings (size of vector representing each token)
embedding_dim = 32
# Number of attention heads in multi-head attention
n_heads = 2
# Number of transformer blocks (layers) in the model
n_layers = 2
# Learning rate for optimizer (how fast the model learns)
lr = 1e-3
# Number of training iterations
epochs = 1500


# Function to sample a random batch of training examples
def sample_training_batch(batch_size=16):
    # Pick random starting positions
    random_indices = torch.randint(len(data) - block_size, (batch_size,))
    # Create input sequences (context tokens)
    input_sequences = torch.stack([data[i:i+block_size] for i in random_indices])
    # Create target sequences (next tokens, shifted by 1)
    target_sequences = torch.stack([data[i+1:i+block_size+1] for i in random_indices])
    # Return input and target batches
    return input_sequences, target_sequences




# Define a GPT-like language model using SentencePiece tokenization
class SentencePieceGPT(nn.Module):
    # Initialize the model components
    def __init__(self):
        # Initialize the parent nn.Module class
        super().__init__()
        # Convert token IDs to dense vectors
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Add position information to tokens
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        # Stack transformer blocks
        self.transformer_blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])

        # Final layer normalization for stability
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        # Output layer: convert embeddings to vocabulary predictions
        self.output_head = nn.Linear(embedding_dim, vocab_size)

    # Convert token indices to embeddings and add position info
    def embed_tokens_and_positions(self, token_indices):
        # Get batch size and sequence length
        batch_size, sequence_length = token_indices.shape
        # Convert token indices to embedding vectors
        token_embeddings = self.token_embedding(token_indices)
        # Get position embeddings
        position_embeddings = self.position_embedding(torch.arange(sequence_length, device=token_indices.device))
        # Add token and position embeddings together
        combined_embeddings = token_embeddings + position_embeddings
        # Return combined embeddings
        return combined_embeddings
    
    # Calculate cross-entropy loss between predictions and targets
    def compute_loss(self, logits, targets):
        # Get dimensions
        batch_size, sequence_length, vocab_size = logits.shape
        # Flatten logits for loss calculation
        logits_flat = logits.view(batch_size * sequence_length, vocab_size)
        # Flatten targets for loss calculation
        targets_flat = targets.view(batch_size * sequence_length)
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        # Return loss value
        return loss
    
    # Forward pass: process input tokens and make predictions
    def forward(self, token_indices, targets=None):
        # Get token and position embeddings
        combined_embeddings = self.embed_tokens_and_positions(token_indices)
        # Process through all transformer blocks (attention + feedforward)
        x = self.transformer_blocks(combined_embeddings)
        # Apply final layer normalization
        x = self.final_layer_norm(x)
        # Convert to raw predictions (scores for each vocabulary token)
        logits = self.output_head(x)
        # Initialize loss as None
        loss = None
        # If we have target tokens (during training)
        if targets is not None:
            # Calculate cross-entropy loss
            loss = self.compute_loss(logits, targets)
        # Return predictions and loss (if calculated)
        return logits, loss

    # Predict logits for the next token
    def predict_next_token_logits(self, token_sequence):
        # Get predictions for all positions
        logits, _ = self(token_sequence)
        # Take predictions from the last position only
        next_token_logits = logits[:, -1, :]
        # Return logits for next token
        return next_token_logits
    
    # Sample the next token from probability distribution
    def sample_next_token(self, logits):
        # Convert logits to probabilities using softmax
        probabilities = F.softmax(logits, dim=-1)
        # Sample next token from probability distribution
        sampled_token = torch.multinomial(probabilities, 1)
        # Return sampled token
        return sampled_token
    
    # Generate new tokens one at a time
    def generate_tokens(self, starting_tokens, max_new_tokens):
        # Start with the initial tokens
        generated_sequence = starting_tokens
        # Repeat for the desired number of new tokens
        for _ in range(max_new_tokens):
            # Use only the last block_size tokens as context
            context_tokens = generated_sequence[:, -block_size:]
            # Get predictions for next token
            next_token_logits = self.predict_next_token_logits(context_tokens)
            # Sample next token from probability distribution
            next_token = self.sample_next_token(next_token_logits)
            # Append new token to the sequence
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
        # Return the complete generated sequence
        return generated_sequence

# Create an instance of the SentencePieceGPT model
model = SentencePieceGPT()
# Create optimizer to update model weights
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training loop: repeat for specified number of epochs
for step in range(epochs):
    # Get a random batch of input and target sequences
    input_batch, target_batch = sample_training_batch()
    # Forward pass: get predictions and calculate loss
    logits, loss = model(input_batch, target_batch)
    # Clear gradients from previous step
    optimizer.zero_grad()
    # Backward pass: calculate gradients
    loss.backward()
    # Update model weights using gradients
    optimizer.step()
    # Print progress every 300 steps
    if step % 300 == 0:
        # Display current step and loss value
        print(f"Step {step}, loss={loss.item():.4f}")

# Import SentencePiece again for text generation
import sentencepiece as spm
# Create processor instance
sp = spm.SentencePieceProcessor()
# Load the trained tokenizer model
sp.load("tokenizer.model")

# Encode starting text "hello" to token IDs
context = torch.tensor([sp.encode("oh carol i am but")], dtype=torch.long)

# Generate 20 new tokens starting from "hello"
out = model.generate_tokens(context, max_new_tokens=20)

# Print header for generated text
print("\nGenerated text:\n")

# Convert generated tensor to list of token IDs
generated_ids = out[0].tolist()
# Decode token IDs back to readable text and print
print(sp.decode(generated_ids))

