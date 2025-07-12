"""
Configuration constants for sentiment analysis explanation system.
"""
from enum import Enum
from typing import List


class SentimentLabel(Enum):
    """Enumeration for sentiment labels."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    LABEL_0 = "LABEL_0"  # Model specific label
    LABEL_1 = "LABEL_1"  # Model specific label


class ModelConfig:
    """Configuration for the sentiment analysis model."""
    DEFAULT_MODEL_NAME = 'distilbert-base-uncased'
    TASK_NAME = 'sentiment-analysis'
    DEVICE_CUDA = "cuda:0"
    DEVICE_CPU = "cpu"


class DatasetConfig:
    """Configuration for dataset handling."""
    DATASET_NAME = 'imdb'
    TEST_SPLIT = 'test'
    DEFAULT_SAMPLE_SIZE = 10


class ExplanationConfig:
    """Configuration for explanation generation."""
    DEFAULT_NUM_FEATURES = 6
    DEFAULT_NUM_SAMPLES = 20
    DEFAULT_SAMPLE_SIZE = 3
    DEFAULT_NUM_EXPLANATIONS = 1
    PROBABILITY_DIMENSIONS = 2
    
    # Class names for explanation display
    CLASS_NAMES: List[str] = [SentimentLabel.POSITIVE.value, SentimentLabel.NEGATIVE.value]


class TestConfig:
    """Configuration for testing explanations."""
    DEFAULT_TEST_SAMPLE_SIZE = 100
    FIDELITY_THRESHOLD = 0.8
    RANDOM_SEED = 42
