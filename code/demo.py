import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_blocks import Block


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
print(text)

# Create vocabulary: get all unique words from the text
words = list(set(text.split()))
print(words)

# Vocabulary size: how many unique words we have
vocab_size = len(words)
print(vocab_size)

# Create mappings: convert words to numbers and numbers back to words
word2idx = {w: i for i, w in enumerate(words)}  # word -> number
print("word2idx : ", word2idx)

idx2word = {i: w for w, i in word2idx.items()}  # number -> word
print("idx2word : ", idx2word) 

# Convert the entire text into a sequence of numbers
data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
print("data : ", data) 
print(len(data))

# Model configuration: these control the size and behavior of the model
block_size = 6      # Maximum number of words the model can look at
embedding_dim = 32  # Size of each word's representation
n_heads = 2         # Number of attention heads
n_layers = 2        # Number of transformer blocks
lr = 1e-3           # Learning rate (how fast the model learns)
epochs = 1500       # How many times to train on the data


def get_batch(batch_size=16):
    """
    Get a random batch of training examples.
    
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
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    # Create input sequences (chunks of block_size words)
    x = torch.stack([data[i:i+block_size] for i in ix])  
    # Create target sequences (same chunks but shifted by 1 word forward)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x, y




class TinyGPT(nn.Module):
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

        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        """
        Process input and make predictions.
        
        This function:
        1. Converts word numbers to embeddings (vectors)
        2. Adds position information
        3. Processes through transformer blocks
        4. Makes predictions about which word comes next
        5. Calculates loss if targets are provided
        
        Args:
            idx: Input tensor with shape (batch, sequence_length) containing word numbers
            targets: Optional target tensor with shape (batch, sequence_length) for training
            
        Returns:
            logits: Predictions for each position, shape (batch, sequence_length, vocab_size)
            loss: The training loss (None if targets not provided)
        """
        B, T = idx.shape 
        # Convert word numbers to vectors
        tok_emb = self.token_embedding(idx) 
        
        # Add position information (where each word is in the sequence)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        # Process through transformer blocks
        x = self.blocks(x) 
        # Final normalization
        x = self.ln_f(x)
        # Convert to word predictions
        logits = self.head(x) 
        loss = None
        # Calculate loss if we have targets (during training)
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new text one word at a time.
        
        This function starts with some input words and keeps adding new words
        by predicting what should come next. It repeats this process until
        it has generated the requested number of new words.
        
        Args:
            idx: Starting sequence with shape (batch, sequence_length)
            max_new_tokens: How many new words to generate
            
        Returns:
            idx: The complete sequence including the original input and all generated words
        """
        for _ in range(max_new_tokens):
            # Only use the last block_size words (model can't see more than that)
            idx_cond = idx[:, -block_size:]
            # Get predictions for what comes next
            logits, _ = self(idx_cond)
            # Look at the last position only (we want to predict the next word)
            logits = logits[:, -1, :]
            # Convert predictions to probabilities
            probs = F.softmax(logits, dim=-1)
            # Randomly sample a word based on the probabilities
            next_idx = torch.multinomial(probs, 1)
            # Add the new word to the sequence
            idx = torch.cat((idx, next_idx), dim=1)
        return idx



# Create the model and optimizer
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training loop: teach the model to predict next words
for step in range(epochs):
    # Get a batch of training examples
    xb, yb = get_batch() 
    # Make predictions and calculate loss
    logits, loss = model(xb, yb)
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
context = torch.tensor([[word2idx["oh"]]], dtype=torch.long)
out = model.generate(context, max_new_tokens=15)

print("\nGenerated text:\n")
print(" ".join(idx2word[int(i)] for i in out[0]))