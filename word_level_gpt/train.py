import sys
import os
# Add parent directory to path to access common folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.transformer_components import Block


# Check PyTorch version and GPU availability
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Training data: a small collection of sentences
corpus = [
    "oh carol i am but a fool",
    "darling i love you",
    "though you treat me cruel",
    "you hurt me and you make me cry",
    "but if you leave me i will surely die",
    "darling there will never be another",
    "cause i love you so",
    "dont ever leave me",
    "say youll never go",
    "i will always want you for my sweetheart",
    "no matter what you do",
    "oh carol im so in love with you"
]

# Add end-of-sentence markers and combine all sentences into one text
corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
# print(text)

# Create vocabulary: get all unique words from the text
words = list(set(text.split()))
# print(words)

# Vocabulary size: how many unique words we have
vocab_size = len(words)
# print(vocab_size)

# Create mappings: convert words to numbers and numbers back to words
word2idx = {w: i for i, w in enumerate(words)}  # word -> number
# print("word2idx : ", word2idx)

idx2word = {i: w for w, i in word2idx.items()}  # number -> word
# print("idx2word : ", idx2word) 

# Convert the entire text into a sequence of numbers
# made one Dimentions tensors because each token is a single number
data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
print("data : ", data) 
# print(len(data))

# Model configuration: these control the size and behavior of the model
block_size = 6      # Maximum number of words the model can look at
embedding_dim = 32  # Size of each word's representation
n_heads = 2         # Number of self attention heads
n_layers = 2        # Number of transformer blocks
lr = 1e-3           # Learning rate (how fast the model learns)
epochs = 1500       # How many times to train on the data


def sample_training_batch(batch_size=16):
    """
    Sample a random batch of training examples from the data.
    
    This function picks random chunks of text from our data and creates
    input-output pairs. The input is a sequence of words, and the output
    is the same sequence shifted by one word (so the model learns to predict
    the next word).
    
    Args:
        batch_size: How many examples to include in the batch
        
    Returns:
        x: Input sequences of shape (batch_size, block_size)
        y: Target sequences of shape (batch_size, block_size) - same as x but shifted by 1
    """
    # Pick random starting positions in the data
    random_indices = torch.randint(len(data) - block_size, (batch_size,))  
    # Create input sequences (chunks of block_size words)
    input_sequences = torch.stack([data[i:i+block_size] for i in random_indices])  
    # Create target sequences (same chunks but shifted by 1 word forward)
    target_sequences = torch.stack([data[i+1:i+block_size+1] for i in random_indices]) 
    return input_sequences, target_sequences




class VerySmallGPT(nn.Module):
    """
    A small GPT-like language model.
    
    This model can learn to predict the next word in a sequence.
    It uses transformer blocks to understand relationships between words
    and generate new text based on what it has learned.
    """
    def __init__(self):
        """
        Set up the model components.
        
        Creates:
        - Token embedding: converts word numbers to vectors
        - Position embedding: tells the model where each word is in the sequence
        - Transformer blocks: the main processing layers
        - Layer norm: helps with training stability
        - Output head: converts the final representation back to word predictions
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 

        self.position_embedding = nn.Embedding(block_size, embedding_dim) # order of the words matter
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def embed_tokens_and_positions(self, token_indices):
        """
        Convert token indices to embeddings and add position information.
        
        Args:
            token_indices: Input tensor with shape (batch, sequence_length) containing word numbers
            
        Returns:
            Combined embeddings with shape (batch, sequence_length, embedding_dim)
        """
        batch_size, sequence_length = token_indices.shape
        # Convert word numbers to vectors
        token_embeddings = self.token_embedding(token_indices) 
        # Add position information (where each word is in the sequence)
        position_embeddings = self.position_embedding(torch.arange(sequence_length, device=token_indices.device))
        # Combine token and position embeddings
        combined_embeddings = token_embeddings + position_embeddings
        return combined_embeddings
    
    def compute_loss(self, logits, targets):
        """
        Calculate the cross-entropy loss between predictions and targets.
        
        Args:
            logits: Model predictions with shape (batch, sequence_length, vocab_size)
            targets: Target token indices with shape (batch, sequence_length)
            
        Returns:
            Loss value (scalar tensor)
        """
        batch_size, sequence_length, vocab_size = logits.shape
        # Reshape for cross-entropy: flatten batch and sequence dimensions
        logits_flat = logits.view(batch_size * sequence_length, vocab_size)
        targets_flat = targets.view(batch_size * sequence_length)
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        return loss
    
    def forward(self, token_indices, targets=None):
        """
        Process input tokens and make predictions about next words.
        
        This function:
        1. Converts word numbers to embeddings (vectors)
        2. Adds position information
        3. Processes through transformer blocks
        4. Makes predictions about which word comes next
        5. Calculates loss if targets are provided
        
        Args:
            token_indices: Input tensor with shape (batch, sequence_length) containing word numbers
            targets: Optional target tensor with shape (batch, sequence_length) for training
            
        Returns:
            logits: Predictions for each position, shape (batch, sequence_length, vocab_size)
            loss: The training loss (None if targets not provided)
        """
        # Embed tokens and add position information
        x = self.embed_tokens_and_positions(token_indices)
        # Process through transformer blocks
        x = self.blocks(x) 
        # Final normalization
        x = self.ln_f(x)
        # Convert to word predictions (logits)
        logits = self.head(x) 
        
        # Calculate loss if we have targets (during training)
        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets)
        return logits, loss

    def predict_next_token(self, token_sequence):
        """
        Predict the next token given a sequence of tokens.
        
        Args:
            token_sequence: Input sequence with shape (batch, sequence_length)
            
        Returns:
            logits: Predictions for the next token, shape (batch, vocab_size)
        """
        # Get predictions for what comes next
        logits, _ = self(token_sequence)
        # Look at the last position only (we want to predict the next word)
        next_token_logits = logits[:, -1, :]
        return next_token_logits
    
    def sample_next_token(self, logits):
        """
        Sample the next token from the probability distribution.
        
        Args:
            logits: Raw predictions with shape (batch, vocab_size)
            
        Returns:
            Sampled token indices with shape (batch, 1)
        """
        # Convert logits to probabilities using softmax
        probabilities = F.softmax(logits, dim=-1)
        # Randomly sample a token based on the probabilities
        sampled_token = torch.multinomial(probabilities, 1)
        return sampled_token
    
    def generate(self, starting_tokens, max_new_tokens):
        """
        Generate new text one token at a time.
        
        This function starts with some input tokens and keeps adding new tokens
        by predicting what should come next. It repeats this process until
        it has generated the requested number of new tokens.
        
        Args:
            starting_tokens: Starting sequence with shape (batch, sequence_length)
            max_new_tokens: How many new tokens to generate
            
        Returns:
            Complete sequence including the original input and all generated tokens
        """
        generated_sequence = starting_tokens
        for _ in range(max_new_tokens):
            # Only use the last block_size tokens (model can't see more than that)
            context_tokens = generated_sequence[:, -block_size:]
            # Predict next token
            next_token_logits = self.predict_next_token(context_tokens)
            # Sample next token from the probability distribution
            next_token = self.sample_next_token(next_token_logits)
            # Add the new token to the sequence
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
        return generated_sequence



# Create the model and optimizer
model = VerySmallGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training loop: teach the model to predict next words
for step in range(epochs):
    # Get a batch of training examples
    input_batch, target_batch = sample_training_batch() 
    # Make predictions and calculate loss
    logits, loss = model(input_batch, target_batch)
    # Reset gradients from previous step
    optimizer.zero_grad()
    # Calculate gradients (how to update the model)
    loss.backward()
    # Update the model weights
    optimizer.step()
    # Print progress every 300 steps
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")


# Generate new text: start with "oh" and see what the model creates
starting_context = torch.tensor([[word2idx["oh"]]], dtype=torch.long)
# max_new_tokens is the number of new words to generate
generated_sequence = model.generate(starting_context, max_new_tokens=15)

print("\nGenerated text:\n")
print(" ".join(idx2word[int(i)] for i in generated_sequence[0]))

