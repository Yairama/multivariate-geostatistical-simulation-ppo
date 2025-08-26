"""
3D CNN Feature Extractor for mining block model data.
Compatible with Stable-Baselines3 custom feature extractors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from typing import Dict, List


class CNN3DFeatureExtractor(BaseFeaturesExtractor):
    """
    3D CNN feature extractor for processing volumetric mining data.
    
    Takes 3D block model data with multiple channels (geological features)
    and extracts spatial features for policy and value networks.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 3, 3],
        strides: List[int] = [1, 2, 2],
        dropout: float = 0.1,
        pooling: str = "adaptive"
    ):
        """
        Initialize 3D CNN feature extractor.
        
        Args:
            observation_space: Gymnasium observation space (Box with shape [C, X, Y, Z])
            features_dim: Dimension of output features
            channels: Number of output channels for each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            strides: Strides for each conv layer
            dropout: Dropout rate
            pooling: Pooling strategy ("adaptive" or "max")
        """
        super().__init__(observation_space, features_dim)
        
        # Extract input dimensions
        n_input_channels = observation_space.shape[0]
        self.input_shape = observation_space.shape[1:]  # (X, Y, Z)
        
        # Build convolutional layers
        conv_layers = []
        in_channels = n_input_channels
        
        for i, (out_channels, kernel_size, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            # Convolutional layer
            conv_layers.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2  # Keep spatial dimensions when stride=1
                )
            )
            
            # Batch normalization
            conv_layers.append(nn.BatchNorm3d(out_channels))
            
            # Activation
            conv_layers.append(nn.ReLU(inplace=True))
            
            # Dropout (except for last layer)
            if i < len(channels) - 1 and dropout > 0:
                conv_layers.append(nn.Dropout3d(dropout))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Pooling layer
        if pooling == "adaptive":
            self.pooling = nn.AdaptiveAvgPool3d((4, 4, 4))  # Fixed output size
            pooled_size = 4 * 4 * 4 * channels[-1]
        elif pooling == "max":
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)
            # Calculate pooled size (approximation)
            pooled_shape = [s // (2 ** len(channels)) for s in self.input_shape]
            pooled_size = pooled_shape[0] * pooled_shape[1] * pooled_shape[2] * channels[-1]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooled_size, features_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D CNN.
        
        Args:
            observations: Tensor of shape [batch_size, channels, x, y, z]
            
        Returns:
            features: Tensor of shape [batch_size, features_dim]
        """
        # Check input shape
        if len(observations.shape) != 5:
            raise ValueError(f"Expected 5D input (batch, channels, x, y, z), got {observations.shape}")
        
        # Forward through conv layers
        x = self.conv_layers(observations)
        
        # Pooling
        x = self.pooling(x)
        
        # Fully connected layers
        features = self.fc_layers(x)
        
        return features


class CNN3DFeatureExtractorSmall(BaseFeaturesExtractor):
    """
    Smaller 3D CNN for faster training/testing with limited computational resources.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # Simpler architecture
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv3d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        
        # Calculate flattened size
        flattened_size = 64 * 2 * 2 * 2  # 512
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, features_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the small 3D CNN."""
        x = self.conv_layers(observations)
        features = self.fc_layers(x)
        return features


class CNN3DFeatureExtractorTiny(BaseFeaturesExtractor):
    """
    Very lightweight 3D CNN for rapid prototyping and testing.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128
    ):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # Minimal architecture
        self.network = nn.Sequential(
            nn.Conv3d(n_input_channels, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(16, features_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


def create_feature_extractor(extractor_type: str, observation_space: gym.Space, **kwargs):
    """
    Factory function to create feature extractors.
    
    Args:
        extractor_type: Type of extractor ("full", "small", "tiny")
        observation_space: Gymnasium observation space
        **kwargs: Additional arguments for the extractor
        
    Returns:
        Feature extractor class
    """
    if extractor_type == "full":
        return CNN3DFeatureExtractor(observation_space, **kwargs)
    elif extractor_type == "small":
        return CNN3DFeatureExtractorSmall(observation_space, **kwargs)
    elif extractor_type == "tiny":
        return CNN3DFeatureExtractorTiny(observation_space, **kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")


if __name__ == "__main__":
    # Test the feature extractors
    import numpy as np
    
    # Create dummy observation space
    obs_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(15, 32, 32, 16),  # 15 channels, 32x32x16 grid
        dtype=np.float32
    )
    
    # Test different extractors
    extractors = {
        "full": CNN3DFeatureExtractor(obs_space, features_dim=512),
        "small": CNN3DFeatureExtractorSmall(obs_space, features_dim=256),
        "tiny": CNN3DFeatureExtractorTiny(obs_space, features_dim=128)
    }
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 15, 32, 32, 16)
    
    print(f"Input shape: {dummy_input.shape}")
    
    for name, extractor in extractors.items():
        print(f"\n{name.upper()} EXTRACTOR:")
        print(f"  Parameters: {sum(p.numel() for p in extractor.parameters()):,}")
        
        with torch.no_grad():
            output = extractor(dummy_input)
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")