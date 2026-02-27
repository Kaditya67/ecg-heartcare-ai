"""
ECG-BERT Model Architecture
A transformer-based model for ECG signal analysis and wave boundary detection.
"""

import torch
import torch.nn as nn
import math


class ECGEmbedding(nn.Module):
    """
    Converts raw ECG signals into embeddings suitable for transformer processing.
    Uses 1D convolution to extract features from the signal.
    """
    def __init__(self, input_channels=1, embedding_dim=768):
        super(ECGEmbedding, self).__init__()
        self.conv = nn.Conv1d(input_channels, embedding_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        x = x.unsqueeze(1)  # [batch_size, 1, sequence_length]
        x = self.conv(x)  # [batch_size, embedding_dim, sequence_length]
        x = x.transpose(1, 2)  # [batch_size, sequence_length, embedding_dim]
        return x


class ECGBERT(nn.Module):
    """
    ECG-BERT: Transformer-based model for ECG wave boundary detection.
    
    Architecture:
    - ECG Signal Embedding Layer
    - Positional Encoding
    - Multi-layer Transformer Encoder
    - Classification Head for wave boundary detection
    
    Args:
        input_channels (int): Number of input channels (default: 1 for single-lead ECG)
        sequence_length (int): Length of ECG signal sequence (default: 2000)
        num_classes (int): Number of wave boundary classes to detect (default: 5)
                          [P onset, P offset, QRS onset, QRS offset, T offset]
        hidden_size (int): Dimension of hidden layers (default: 128)
        num_layers (int): Number of transformer encoder layers (default: 4)
        num_attention_heads (int): Number of attention heads (default: 4)
        intermediate_size (int): Dimension of feedforward network (default: 512)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        input_channels=1,
        sequence_length=2000,
        num_classes=5,
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        dropout=0.1
    ):
        super(ECGBERT, self).__init__()
        
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # ECG embedding layer
        self.embedding = ECGEmbedding(input_channels, hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(sequence_length, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Classification head for wave boundary detection
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        """
        Forward pass of ECG-BERT model.
        
        Args:
            x (torch.Tensor): Input ECG signal of shape [batch_size, sequence_length]
        
        Returns:
            torch.Tensor: Wave boundary predictions of shape [batch_size, sequence_length, num_classes]
        """
        batch_size, seq_length = x.shape
        
        # Create ECG embeddings
        embeddings = self.embedding(x)  # [batch_size, sequence_length, hidden_size]
        
        # Add position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer encoder layers
        encoder_output = embeddings
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        
        # Classification
        logits = self.classifier(encoder_output)  # [batch_size, sequence_length, num_classes]
        
        return logits
    
    def predict_boundaries(self, x, threshold=0.5):
        """
        Predict wave boundaries from ECG signal.
        
        Args:
            x (torch.Tensor): Input ECG signal
            threshold (float): Probability threshold for boundary detection
        
        Returns:
            dict: Dictionary containing detected boundary indices for each wave type
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            
            # Get boundary predictions for each class
            predictions = (probabilities > threshold).cpu().numpy()
            
            boundaries = {
                'P_onsets': [],
                'P_offsets': [],
                'QRS_onsets': [],
                'QRS_offsets': [],
                'T_offsets': []
            }
            
            boundary_names = ['P_onsets', 'P_offsets', 'QRS_onsets', 'QRS_offsets', 'T_offsets']
            
            for batch_idx in range(predictions.shape[0]):
                for class_idx, boundary_name in enumerate(boundary_names):
                    indices = predictions[batch_idx, :, class_idx].nonzero()[0]
                    boundaries[boundary_name].append(indices.tolist())
            
            return boundaries


def create_ecgbert_model(config=None):
    """
    Factory function to create ECG-BERT model with configuration.
    
    Args:
        config (dict): Configuration dictionary with model parameters
    
    Returns:
        ECGBERT: Initialized ECG-BERT model
    """
    if config is None:
        config = {
            'input_channels': 1,
            'sequence_length': 2000,
            'num_classes': 5,
            'hidden_size': 128,
            'num_layers': 4,
            'num_attention_heads': 4,
            'intermediate_size': 512,
            'dropout': 0.1
        }
    
    model = ECGBERT(**config)
    return model
