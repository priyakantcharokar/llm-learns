# LLM Learns - A Simple Transformer Language Model

A minimal implementation of a GPT-like language model using PyTorch. This project demonstrates how transformers work by building a small model that can learn to generate text from training data (in this case, "Oh Carol" song lyrics).

## 📚 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Important Terms](#important-terms)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Components Explained](#components-explained)

## 🎯 Overview

This project implements a very small GPT (Generative Pre-trained Transformer) model from scratch. The model learns to predict the next word in a sequence by training on a small corpus of text. After training, it can generate new text that follows similar patterns to what it learned.

**What this project teaches:**
- How transformer architecture works
- Self-attention mechanism
- Multi-head attention
- Language modeling basics
- Text generation

## 📁 Project Structure

```
llm-learns/
├── code/
│   ├── transformer_components.py  # Core transformer components
│   └── train.py                   # Training and generation script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔑 Important Terms

### **Transformer**
A type of neural network architecture that uses attention mechanisms to understand relationships between words in a sequence. Transformers are the foundation of modern language models like GPT, BERT, and ChatGPT.

### **Attention / Self-Attention**
A mechanism that allows the model to look at all words in a sequence and decide which ones are most relevant to each other. Think of it as the model "paying attention" to different parts of the input when processing each word.

### **Embedding**
Converting words (which are text) into numbers (vectors) that the computer can work with. Each word gets represented as a list of numbers that capture its meaning.

### **Token**
A unit of text - could be a word, part of a word, or a punctuation mark. In this project, we use words as tokens.

### **Vocabulary**
The complete set of unique words that the model knows about. Every word in the training data becomes part of the vocabulary.

### **Block Size / Context Window**
The maximum number of words the model can look at at once. If block_size is 6, the model can only see the last 6 words when making predictions.

### **Multi-Head Attention**
Using multiple attention mechanisms in parallel. Each "head" can learn to focus on different types of relationships between words (e.g., one head might focus on grammar, another on meaning).

### **Feed Forward Network**
A simple neural network that processes each word independently after attention. It adds complexity and helps the model learn more patterns.

### **Layer Normalization**
A technique that helps stabilize training by normalizing the values in each layer. It makes the model learn faster and more reliably.

### **Skip Connection / Residual Connection**
A connection that allows information to flow directly through a layer without modification. This helps the model learn better by preventing information loss.

### **Logits**
The raw predictions from the model before they're converted to probabilities. Higher logit values mean the model thinks that word is more likely.

### **Cross-Entropy Loss**
A way to measure how wrong the model's predictions are. The model tries to minimize this during training.

### **Learning Rate**
How big of steps the model takes when learning. Too high = unstable learning, too low = very slow learning.

### **Epoch**
One complete pass through all the training data. The model sees all examples once per epoch.

### **Batch**
A small group of training examples processed together. Instead of learning from one example at a time, the model learns from multiple examples simultaneously.

### **Softmax**
A mathematical function that converts logits (raw scores) into probabilities. It ensures all probabilities add up to 1.0, making it easier to pick the most likely word.

### **Optimizer (AdamW)**
An algorithm that updates the model's weights during training. AdamW is an improved version of Adam that helps the model learn more effectively by adjusting how big steps to take based on past gradients.

### **Gradient**
The direction and magnitude of how to change the model's weights to reduce the loss. Think of it as a compass pointing toward better predictions.

### **Backpropagation**
The process of calculating gradients by working backwards through the network. It figures out how each part of the model contributed to the error.

### **Keys, Queries, and Values (KQV)**
The three components used in attention:
- **Query**: "What am I looking for?"
- **Key**: "What do I have to offer?"
- **Value**: "What information do I contain?"

The attention mechanism uses these to determine how much each word should pay attention to others.

### **Causal Masking**
A technique that prevents the model from seeing future words when making predictions. This ensures the model can only use information from previous words, which is crucial for text generation.

### **Position Embedding**
Additional information added to word embeddings that tells the model where each word appears in the sequence. This is important because word order matters in language (e.g., "dog bites man" vs "man bites dog").

### **Token Embedding**
The process of converting word indices (numbers) into dense vectors (lists of numbers) that capture semantic meaning. Each word gets a unique vector representation.

### **ReLU (Rectified Linear Unit)**
An activation function that turns negative values to zero and keeps positive values unchanged. It adds non-linearity to the network, allowing it to learn complex patterns.

### **Linear Layer**
A basic neural network layer that applies a linear transformation (multiply by weights and add bias). It's the building block of most neural networks.

## 🧠 How It Works

### 1. **Data Preparation**
- The training text (song lyrics) is split into words
- Each unique word gets assigned a number (tokenization)
- The text is converted into sequences of numbers

### 2. **Model Architecture**
The model consists of:
- **Token Embedding**: Converts word numbers to vectors
- **Position Embedding**: Tells the model where each word is in the sequence
- **Transformer Blocks**: Multiple layers that process the text using attention
- **Output Head**: Converts the final representation back to word predictions

### 3. **Training Process**
1. Get a random batch of text chunks (input sequences and their corresponding next words)
2. Feed input sequences to the model
3. Model makes predictions about what word should come next
4. Compare predictions with actual next words
5. Calculate loss using cross-entropy (measures prediction error)
6. Use backpropagation to calculate gradients
7. Optimizer updates model weights to reduce loss
8. Repeat thousands of times until the model learns the patterns

### 4. **Text Generation**
1. Start with a seed word (e.g., "oh")
2. Model predicts what word should come next
3. Add that word to the sequence
4. Use the last few words to predict the next one
5. Repeat until desired length

## 🚀 Installation

### Prerequisites
- Python 3.10 or later
- pip (Python package manager)

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/priyakantcharokar/llm-learns.git
cd llm-learns
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

This will install:
- `torch` - PyTorch deep learning framework
- `torchvision` - Additional PyTorch utilities
- `torchaudio` - Audio processing (included with PyTorch)

## 💻 Usage

### Running the Demo

Navigate to the code directory and run:

```bash
cd code
python train.py
```

### What Happens

1. **Setup**: The script checks your PyTorch installation and GPU availability
2. **Data Processing**: Converts "Oh Carol" lyrics into a format the model can learn from
3. **Model Creation**: Builds a small transformer model
4. **Training**: Trains the model for 1500 steps, printing loss every 300 steps
5. **Generation**: Generates new text starting with "oh"

### Expected Output

You'll see:
- PyTorch version and GPU info
- Training progress (loss decreasing over time)
- Generated text based on the learned patterns

Example output:
```
Step 0, loss=3.4521
Step 300, loss=2.1234
Step 600, loss=1.5678
...
Generated text:
oh carol i am but a fool darling i love you so...
```

## 🧩 Components Explained

### `transformer_components.py`

Contains the core building blocks of the transformer:

#### **SelfAttentionHead**
- A single attention mechanism
- Uses keys, queries, and values to compute relationships
- Implements causal masking (can't see future words)

#### **MultiHeadAttention**
- Runs multiple attention heads in parallel
- Each head learns different relationships
- Combines all heads into a single output

#### **FeedForward**
- A simple 2-layer neural network
- Expands then contracts the representation
- Adds non-linearity with ReLU activation

#### **Block**
- A complete transformer layer
- Combines attention and feed-forward
- Uses layer normalization and skip connections

### `train.py`

The main training and generation script:

#### **VerySmallGPT Class**
A minimal GPT-like model that:
- Embeds tokens and positions
- Processes through transformer blocks
- Predicts next words

#### **Key Functions**

**`sample_training_batch(batch_size=16)`**
- Randomly selects chunks of text from the training data
- Creates input sequences and target sequences
- Target sequences are shifted by one position (next word prediction)
- Returns batches ready for training

**`forward(token_indices, targets=None)`**
- Main processing function that runs when you call the model
- Converts word indices to embeddings
- Adds position information
- Processes through transformer blocks
- Converts to word predictions (logits)
- Calculates loss if targets are provided

**`generate(starting_tokens, max_new_tokens)`**
- Text generation function
- Starts with a seed word or phrase
- Predicts next token using softmax probabilities
- Samples from the probability distribution
- Appends new token and repeats until desired length

## 📊 Model Configuration

The current model settings (in `train.py`):

```python
block_size = 6      # Context window: 6 words
embedding_dim = 32  # Word representation size: 32 numbers
n_heads = 2         # Attention heads: 2
n_layers = 2        # Transformer blocks: 2
lr = 1e-3          # Learning rate: 0.001
epochs = 1500      # Training steps: 1500
```

**You can experiment with these values:**
- Increase `embedding_dim` for richer word representations (try 64, 128)
- Add more `n_heads` for more diverse attention patterns (try 4, 8)
- Increase `n_layers` for deeper understanding (try 4, 6)
- Adjust `lr` if training is unstable (too high) or too slow (too low)
- Increase `block_size` to allow longer context (try 8, 12)
- Adjust `epochs` based on when loss stops decreasing

**Note**: Larger models need more training data and take longer to train!

## 🔍 Code Walkthrough

### Data Flow Example

1. **Input**: `"oh carol i am but"` → Converted to numbers: `[5, 2, 8, 1, 3]`
2. **Embedding**: Each number becomes a 32-dimensional vector
3. **Position**: Position embeddings added (word 0, word 1, word 2, etc.)
4. **Attention**: Model looks at relationships between words
5. **Feed Forward**: Each word processed independently
6. **Output**: Predictions for next word (probabilities for each word in vocabulary)
7. **Generation**: Sample from probabilities to get next word

### Training Step Breakdown

```python
# 1. Get training batch
input_batch, target_batch = sample_training_batch()  # input = sequences, target = next words

# 2. Forward pass
logits, loss = model(input_batch, target_batch)  # Model makes predictions

# 3. Backward pass
loss.backward()  # Calculate gradients

# 4. Update weights
optimizer.step()  # Adjust model parameters
```

## 🎓 Learning Resources

This project is designed to help you understand:
- How transformers process sequential data
- The role of attention in language models
- How language models learn to predict text
- The basics of neural network training
- The relationship between keys, queries, and values
- Why position embeddings are necessary
- How text generation works step-by-step

## 🤝 Contributing

Feel free to experiment with:
- Different training data
- Model architecture modifications
- Hyperparameter tuning
- Adding new features

## 📝 License

This is an educational project. Feel free to use and modify as needed.

## 🙏 Acknowledgments

This implementation is inspired by:
- The original Transformer paper: "Attention Is All You Need"
- Andrej Karpathy's nanoGPT
- Various educational resources on transformers

---

**Happy Learning! 🚀**

