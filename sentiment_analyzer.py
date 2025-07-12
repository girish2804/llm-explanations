"""
Core sentiment analysis functionality.
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from transformers import pipeline
import logging

from config import ModelConfig, SentimentLabel, ExplanationConfig


class SentimentAnalyzer:
    """Handles sentiment analysis using pre-trained models."""
    
    def __init__(self, model_name: str = ModelConfig.DEFAULT_MODEL_NAME):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the model to use for sentiment analysis
        """
        self.device = self._get_device()
        self.model_name = model_name
        self.classifier = self._initialize_classifier()
        
    def _get_device(self) -> str:
        """Determine the appropriate device for computation."""
        return ModelConfig.DEVICE_CUDA if torch.cuda.is_available() else ModelConfig.DEVICE_CPU
        
    def _initialize_classifier(self) -> pipeline:
        """Initialize the sentiment classification pipeline."""
        try:
            return pipeline(
                ModelConfig.TASK_NAME,
                model=self.model_name,
                device=0 if self.device == ModelConfig.DEVICE_CUDA else -1
            )
        except Exception as e:
            logging.error(f"Failed to initialize classifier: {e}")
            raise
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text for sentiment analysis
            
        Returns:
            Dictionary containing prediction results
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        try:
            return self.classifier(text)
        except Exception as e:
            logging.error(f"Prediction failed for text: {e}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input must be a list of non-empty strings")
            
        try:
            return [self.classifier(text) for text in texts]
        except Exception as e:
            logging.error(f"Batch prediction failed: {e}")
            raise
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for texts (compatible with LIME).
        
        Args:
            texts: List of input texts
            
        Returns:
            Numpy array of shape (n_samples, n_classes) with probabilities
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        batch_size = len(texts)
        probabilities = np.empty((batch_size, ExplanationConfig.PROBABILITY_DIMENSIONS))
        
        try:
            for idx, text in enumerate(texts):
                prediction = self.classifier(text)
                score = prediction[0]['score']
                complement_score = 1.0 - score
                
                # Handle different label formats
                if prediction[0]['label'] == SentimentLabel.LABEL_1.value:
                    # LABEL_1 typically means positive
                    probabilities[idx] = np.array([score, complement_score])
                else:
                    # LABEL_0 typically means negative
                    probabilities[idx] = np.array([complement_score, score])
                    
        except Exception as e:
            logging.error(f"Probability prediction failed: {e}")
            raise
            
        return probabilities
    
    def get_device_info(self) -> str:
        """Get current device information."""
        return self.device
