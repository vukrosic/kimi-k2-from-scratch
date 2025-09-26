# Natural Language Processing with PyTorch

## Learning Objectives
- Master text preprocessing and tokenization
- Understand word embeddings and sequence models
- Learn about transformers and attention mechanisms
- Practice with real NLP tasks

## Text Preprocessing

### Basic Text Processing
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, min_freq=2, max_vocab_size=10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        # Use NLTK tokenizer
        tokens = word_tokenize(text)
        
        # Remove stop words and apply stemming
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize(cleaned_text)
            word_counts.update(tokens)
        
        # Filter by frequency and size
        vocab_words = [word for word, count in word_counts.most_common(self.max_vocab_size)
                      if count >= self.min_freq]
        
        # Add special tokens
        vocab_words = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + vocab_words
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab = set(vocab_words)
        
        return self.vocab
    
    def text_to_sequence(self, text, max_length=None):
        """Convert text to sequence of indices"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        
        # Convert to indices
        sequence = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) 
                   for token in tokens]
        
        # Pad or truncate to max_length
        if max_length:
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            else:
                sequence.extend([self.word_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        return sequence

# Example usage
texts = [
    "This is a sample text for NLP processing.",
    "Natural language processing is fascinating!",
    "PyTorch makes deep learning accessible."
]

preprocessor = TextPreprocessor()
vocab = preprocessor.build_vocab(texts)
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample words: {list(vocab)[:10]}")

# Convert text to sequence
sequence = preprocessor.text_to_sequence(texts[0], max_length=20)
print(f"Sequence: {sequence}")
```

### Advanced Text Processing
```python
import spacy
from transformers import AutoTokenizer

# Using spaCy for advanced NLP
class AdvancedTextProcessor:
    def __init__(self, model_name='en_core_web_sm'):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Please install spaCy model: python -m spacy download {model_name}")
            self.nlp = None
    
    def extract_features(self, text):
        """Extract linguistic features"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        
        features = {
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [token.pos_ for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'sentences': [sent.text for sent in doc.sents],
            'dependency_parse': [(token.text, token.dep_, token.head.text) for token in doc]
        }
        
        return features
    
    def get_word_embeddings(self, text):
        """Get word embeddings using spaCy"""
        if not self.nlp:
            return None
        
        doc = self.nlp(text)
        embeddings = [token.vector for token in doc if token.has_vector]
        
        return torch.tensor(embeddings) if embeddings else None

# Using Hugging Face tokenizers
class HuggingFaceProcessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_text(self, text, max_length=512):
        """Tokenize text using Hugging Face tokenizer"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encoding
    
    def batch_tokenize(self, texts, max_length=512):
        """Tokenize batch of texts"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encodings

# Example usage
hf_processor = HuggingFaceProcessor()
text = "This is a sample text for tokenization."
encoding = hf_processor.tokenize_text(text)
print(f"Input IDs: {encoding['input_ids']}")
print(f"Attention Mask: {encoding['attention_mask']}")
```

## Word Embeddings

### Word2Vec Implementation
```python
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        
        # Input and output embeddings
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self.init_embeddings()
    
    def init_embeddings(self):
        """Initialize embeddings with uniform distribution"""
        init_range = 0.5 / self.embedding_dim
        self.input_embeddings.weight.data.uniform_(-init_range, init_range)
        self.output_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, center_words, context_words, negative_words=None):
        """Forward pass for Skip-gram model"""
        # Get embeddings
        center_emb = self.input_embeddings(center_words)  # [batch_size, embedding_dim]
        context_emb = self.output_embeddings(context_words)  # [batch_size, embedding_dim]
        
        # Positive score
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # [batch_size]
        pos_loss = -F.logsigmoid(pos_score).mean()
        
        # Negative sampling loss
        if negative_words is not None:
            neg_emb = self.output_embeddings(negative_words)  # [batch_size, num_neg, embedding_dim]
            neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # [batch_size, num_neg]
            neg_loss = -F.logsigmoid(-neg_score).mean()
            
            return pos_loss + neg_loss
        
        return pos_loss

# Training Word2Vec
def train_word2vec(model, data_loader, num_epochs=5, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (center, context, negative) in enumerate(data_loader):
            optimizer.zero_grad()
            
            loss = model(center, context, negative)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch} completed. Average Loss: {total_loss/len(data_loader):.4f}')
```

### GloVe Implementation
```python
class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVe, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Bias terms
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)
        
        # Initialize parameters
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize parameters"""
        init_range = 0.5 / self.embedding_dim
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
        self.word_biases.weight.data.zero_()
        self.context_biases.weight.data.zero_()
    
    def forward(self, word_ids, context_ids, cooccurrence_counts):
        """Forward pass for GloVe"""
        # Get embeddings and biases
        word_emb = self.word_embeddings(word_ids)
        context_emb = self.context_embeddings(context_ids)
        word_bias = self.word_biases(word_ids).squeeze()
        context_bias = self.context_biases(context_ids).squeeze()
        
        # Compute dot product
        dot_product = torch.sum(word_emb * context_emb, dim=1)
        
        # Compute loss
        log_counts = torch.log(cooccurrence_counts.float())
        prediction = dot_product + word_bias + context_bias
        
        # Weighted MSE loss
        weights = torch.clamp(cooccurrence_counts.float() / 100.0, max=1.0)
        loss = weights * (prediction - log_counts) ** 2
        
        return loss.mean()

# Training GloVe
def train_glove(model, data_loader, num_epochs=10, learning_rate=0.05):
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (word_ids, context_ids, counts) in enumerate(data_loader):
            optimizer.zero_grad()
            
            loss = model(word_ids, context_ids, counts)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch} completed. Average Loss: {total_loss/len(data_loader):.4f}')
```

## Sequence Models

### LSTM for Text Classification
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x, lengths=None):
        # Embedding
        embedded = self.embedding(x)
        
        # Pack sequence if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Unpack sequence
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Use last hidden state
        # For bidirectional LSTM, concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Apply dropout and classification
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output

# Training function
def train_lstm_classifier(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target, lengths) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data, lengths)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target, lengths in val_loader:
                output = model(data, lengths)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Print epoch results
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Epoch {epoch}: Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
```

### GRU for Sequence Generation
```python
class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(GRULanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # GRU
        output, hidden = self.gru(embedded, hidden)
        output = self.dropout(output)
        
        # Output projection
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
    
    def generate(self, start_token, max_length=100, temperature=1.0):
        """Generate text sequence"""
        self.eval()
        generated = [start_token]
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([[generated[-1]]])
                output, hidden = self.forward(input_tensor, hidden)
                
                # Apply temperature
                output = output / temperature
                
                # Sample from distribution
                probs = F.softmax(output[0, -1], dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                # Stop if end token
                if next_token == 0:  # Assuming 0 is end token
                    break
        
        return generated

# Training function for language model
def train_language_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(data)
            
            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch} completed. Average Loss: {total_loss/len(train_loader):.4f}')
        
        # Generate sample text
        sample = model.generate(start_token=1, max_length=50, temperature=0.8)
        print(f"Generated sample: {sample}")
```

## Transformers and Attention

### Self-Attention Mechanism
```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Attention calculation
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        # Add & Norm
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, src_vocab_size)
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        out = self.fc_out(out)
        return out
```

## Practice Exercises

### Exercise 1: Sentiment Analysis
Build a sentiment analysis model that:
- Classifies movie reviews as positive or negative
- Uses LSTM or Transformer architecture
- Implements proper text preprocessing
- Achieves >85% accuracy

### Exercise 2: Text Generation
Create a text generation model that:
- Generates coherent text sequences
- Uses GRU or Transformer architecture
- Implements temperature sampling
- Generates creative and diverse text

### Exercise 3: Named Entity Recognition
Implement an NER model that:
- Identifies entities in text
- Uses bidirectional LSTM
- Implements CRF layer for sequence labeling
- Evaluates on standard NER datasets

### Exercise 4: Question Answering
Build a QA system that:
- Answers questions based on context
- Uses attention mechanisms
- Implements proper evaluation metrics
- Handles different question types

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning Natural Language Processing with PyTorch. I understand basic neural networks but I'm struggling with:

1. Text preprocessing and tokenization techniques
2. Word embeddings and their applications
3. Sequence models like LSTM and GRU
4. Attention mechanisms and transformers
5. Building NLP models for different tasks
6. Handling variable-length sequences and padding

Please:
- Explain each concept with practical examples
- Show me how to preprocess text data effectively
- Help me understand the mathematics behind embeddings and attention
- Walk me through building different NLP architectures
- Give me exercises to practice with real text data
- Explain common challenges and solutions in NLP

I want to build effective NLP models for real-world applications. Please provide hands-on examples and help me understand the design principles."

## Key Takeaways
- Text preprocessing is crucial for NLP model performance
- Word embeddings capture semantic relationships between words
- Sequence models are essential for handling sequential text data
- Attention mechanisms allow models to focus on relevant parts of input
- Transformers have revolutionized NLP with their parallel processing capabilities
- Proper evaluation metrics are important for different NLP tasks

## Next Steps
Master NLP with PyTorch and you'll be ready for:
- Advanced transformer architectures
- Large language models
- Multimodal models
- Conversational AI
- Machine translation
- Text summarization
