"""
Document Processing Deep Learning Model

PyTorch LSTM-based text classifier for document type classification with:
- Word-level tokenizer
- 100-dim learned embeddings
- Bidirectional LSTM (128 hidden units)
- FC layer for classification
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import joblib
from collections import Counter
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class Tokenizer:
    """Simple word-level tokenizer."""

    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        """
        Initialize tokenizer.

        Args:
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum word frequency to include in vocabulary
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)

        # Filter by minimum frequency and limit vocabulary size
        filtered_words = [
            word for word, count in word_counts.most_common(self.max_vocab_size)
            if count >= self.min_freq
        ]

        # Build vocabulary
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        for idx, word in enumerate(filtered_words, start=2):
            self.word2idx[word] = idx

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Convert text to token indices."""
        tokens = self._tokenize(text)
        indices = [
            self.word2idx.get(token, self.word2idx[self.unk_token])
            for token in tokens
        ]

        if max_length:
            indices = indices[:max_length]

        return indices

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        """Convert batch of texts to token indices."""
        return [self.encode(text, max_length) for text in texts]

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)


class DocumentClassifierNetwork(nn.Module):
    """Bidirectional LSTM network for document classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize the network.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: LSTM hidden dimension
            num_classes: Number of document classes
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of token indices (batch_size, max_seq_len)
            lengths: Actual lengths of sequences (batch_size,)

        Returns:
            Output logits (batch_size, num_classes)
        """
        # Embed tokens
        embedded = self.embedding(x)  # (batch_size, max_seq_len, embedding_dim)

        # Pack padded sequence for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed)

        # Use final hidden state
        # For bidirectional: concatenate forward and backward final states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # Classification
        output = self.fc(hidden)

        return output


class DocumentClassifierDeep:
    """Document type classifier using PyTorch LSTM."""

    def __init__(
        self,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        max_vocab_size: int = 10000,
        max_seq_length: int = 512,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 30,
        random_state: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize the deep document classifier.

        Args:
            embedding_dim: Dimension of word embeddings
            hidden_dim: LSTM hidden dimension
            max_vocab_size: Maximum vocabulary size
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            random_state: Random seed
            device: Device to use ('cuda' or 'cpu')
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.tokenizer = Tokenizer(max_vocab_size=max_vocab_size)
        self.label2idx = {}
        self.idx2label = {}
        self.document_types = ['invoice', 'bank_statement', 'tax_form', 'contract']

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _prepare_data(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        fit_tokenizer: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare text data for the model.

        Args:
            texts: List of document texts
            labels: Optional list of document type labels
            fit_tokenizer: Whether to fit the tokenizer

        Returns:
            Tuple of (padded_sequences, lengths, labels_tensor)
        """
        # Fit or use tokenizer
        if fit_tokenizer:
            self.tokenizer.fit(texts)

        # Encode texts
        encoded = self.tokenizer.encode_batch(texts, max_length=self.max_seq_length)

        # Get lengths
        lengths = torch.tensor([len(seq) for seq in encoded])

        # Pad sequences
        padded = [torch.tensor(seq) for seq in encoded]
        padded = pad_sequence(padded, batch_first=True, padding_value=0)

        # Encode labels
        labels_tensor = None
        if labels is not None:
            if fit_tokenizer:
                unique_labels = sorted(set(labels))
                self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
                self.idx2label = {idx: label for label, idx in self.label2idx.items()}

            label_indices = [self.label2idx[label] for label in labels]
            labels_tensor = torch.tensor(label_indices)

        return padded.to(self.device), lengths.to(self.device), labels_tensor

    def train(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        label_column: str = 'document_type',
        test_size: float = 0.2,
        early_stopping_patience: int = 5
    ) -> Dict[str, Any]:
        """
        Train the document classifier.

        Args:
            df: Training DataFrame
            text_column: Column containing document text
            label_column: Column containing document type labels
            test_size: Proportion of test set
            early_stopping_patience: Epochs to wait for improvement

        Returns:
            Dictionary with training metrics
        """
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()

        # Split data
        texts_train, texts_test, labels_train, labels_test = train_test_split(
            texts, labels, test_size=test_size, random_state=self.random_state, stratify=labels
        )

        # Prepare training data
        X_train, lengths_train, y_train = self._prepare_data(
            texts_train, labels_train, fit_tokenizer=True
        )
        y_train = y_train.to(self.device)

        # Prepare test data
        X_test, lengths_test, y_test = self._prepare_data(texts_test, labels_test)
        y_test = y_test.to(self.device)

        # Initialize model
        num_classes = len(self.label2idx)
        self.model = DocumentClassifierNetwork(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            dropout=self.dropout
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create DataLoader
        train_dataset = TensorDataset(X_train, lengths_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, lengths_batch, y_batch in train_loader:
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(X_batch, lengths_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test, lengths_test)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
            y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()

        y_test_np = y_test.cpu().numpy()

        # Convert indices back to labels for report
        y_pred_labels = [self.idx2label[idx] for idx in y_pred]
        y_test_labels = [self.idx2label[idx] for idx in y_test_np]

        metrics = {
            'accuracy': accuracy_score(y_test_np, y_pred),
            'precision_weighted': precision_score(y_test_np, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test_np, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test_np, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test_np, y_pred).tolist(),
            'classification_report': classification_report(
                y_test_labels, y_pred_labels, zero_division=0
            ),
            'epochs_trained': epoch + 1,
            'final_train_loss': train_losses[-1],
            'vocab_size': self.tokenizer.vocab_size,
            'num_classes': num_classes
        }

        return metrics

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict document types.

        Args:
            texts: List of document texts

        Returns:
            List of predicted document types
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        proba = self.predict_proba(texts)
        predictions = proba.argmax(axis=1)

        return [self.idx2label[idx] for idx in predictions]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict document type probabilities.

        Args:
            texts: List of document texts

        Returns:
            Array of probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare data
        X, lengths, _ = self._prepare_data(texts)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X, lengths)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def classify_document(self, text: str) -> Tuple[str, float]:
        """
        Classify a single document.

        Args:
            text: Document text

        Returns:
            Tuple of (document_type, confidence)
        """
        proba = self.predict_proba([text])[0]
        predicted_idx = proba.argmax()
        confidence = proba[predicted_idx]

        return self.idx2label[predicted_idx], confidence

    def save(self, filepath: str):
        """Save the model."""
        model_state = self.model.state_dict() if self.model else None

        joblib.dump({
            'model_state': model_state,
            'tokenizer': self.tokenizer,
            'label2idx': self.label2idx,
            'idx2label': self.idx2label,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'max_seq_length': self.max_seq_length
        }, filepath)

    def load(self, filepath: str):
        """Load the model."""
        data = joblib.load(filepath)

        self.tokenizer = data['tokenizer']
        self.label2idx = data['label2idx']
        self.idx2label = data['idx2label']
        self.embedding_dim = data['embedding_dim']
        self.hidden_dim = data['hidden_dim']
        self.dropout = data['dropout']
        self.max_seq_length = data['max_seq_length']

        # Rebuild and load model
        if data['model_state'] is not None:
            num_classes = len(self.label2idx)
            self.model = DocumentClassifierNetwork(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                num_classes=num_classes,
                dropout=self.dropout
            ).to(self.device)
            self.model.load_state_dict(data['model_state'])
            self.model.eval()


def main():
    """Example usage of the deep document classifier."""
    from src.data_generators.document_processing_generator import DocumentProcessingDataGenerator

    # Generate data
    print("Generating synthetic documents...")
    generator = DocumentProcessingDataGenerator(random_state=42)
    df = generator.generate(n_samples=1000)

    print(f"Document type distribution:")
    print(df['document_type'].value_counts())

    # Train model
    print("\nTraining deep learning document classifier...")
    model = DocumentClassifierDeep(
        embedding_dim=100,
        hidden_dim=128,
        max_vocab_size=10000,
        max_seq_length=512,
        dropout=0.3,
        epochs=30,
        batch_size=32,
        random_state=42
    )
    metrics = model.train(df, text_column='text', label_column='document_type')

    # Display metrics
    print("\nModel Performance Metrics:")
    print("=" * 50)
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    print("\nClassification Report:")
    print(metrics['classification_report'])

    # Test single document classification
    print("\n" + "=" * 50)
    print("Single Document Classification Example:")
    print("=" * 50)
    sample_doc = df.iloc[0]
    doc_type, confidence = model.classify_document(sample_doc['text'])
    print(f"Actual type: {sample_doc['document_type']}")
    print(f"Predicted type: {doc_type}")
    print(f"Confidence: {confidence:.2%}")

    # Save model
    model.save("models/document_processing/document_classifier_deep.pkl")
    print("\nModel saved to models/document_processing/document_classifier_deep.pkl")


if __name__ == "__main__":
    main()
