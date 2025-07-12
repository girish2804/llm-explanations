"""
Explanation generation using LIME and SHAP.
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
import shap

from config import ExplanationConfig
from sentiment_analyzer import SentimentAnalyzer


class ExplanationGenerator:
    """Generates explanations for sentiment analysis predictions."""
    
    def __init__(self, analyzer: SentimentAnalyzer):
        """
        Initialize the explanation generator.
        
        Args:
            analyzer: Sentiment analyzer instance
        """
        self.analyzer = analyzer
        self.lime_explainer = LimeTextExplainer(
            class_names=ExplanationConfig.CLASS_NAMES
        )
        self.shap_explainer = None
        
    def _initialize_shap_explainer(self, background_texts: List[str]) -> None:
        """
        Initialize SHAP explainer with background data.
        
        Args:
            background_texts: Background texts for SHAP explanation
        """
        try:
            # Use a subset of background texts to avoid memory issues
            background_sample = background_texts[:min(len(background_texts), 50)]
            
            # Create a wrapper function for SHAP
            def prediction_wrapper(texts):
                if isinstance(texts, str):
                    texts = [texts]
                elif isinstance(texts, np.ndarray):
                    texts = texts.tolist()
                
                probabilities = self.analyzer.predict_proba(texts)
                return probabilities
            
            self.shap_explainer = shap.Explainer(
                prediction_wrapper,
                background_sample,
                max_evals=100
            )
            
        except Exception as e:
            logging.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def explain_instance_lime(self, 
                             text: str, 
                             num_features: int = ExplanationConfig.DEFAULT_NUM_FEATURES,
                             num_samples: int = ExplanationConfig.DEFAULT_NUM_SAMPLES) -> Any:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            text: Text to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME
            
        Returns:
            LIME explanation object
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        try:
            explanation = self.lime_explainer.explain_instance(
                text,
                self.analyzer.predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )
            return explanation
        except Exception as e:
            logging.error(f"LIME explanation failed: {e}")
            raise
    
    def explain_instance_shap(self, 
                             text: str,
                             background_texts: Optional[List[str]] = None) -> Optional[Any]:
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            text: Text to explain
            background_texts: Background texts for SHAP (if not already initialized)
            
        Returns:
            SHAP explanation object or None if SHAP is not available
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        if self.shap_explainer is None:
            if background_texts:
                self._initialize_shap_explainer(background_texts)
            else:
                logging.warning("SHAP explainer not initialized and no background texts provided")
                return None
        
        if self.shap_explainer is None:
            return None
            
        try:
            shap_values = self.shap_explainer([text])
            return shap_values
        except Exception as e:
            logging.error(f"SHAP explanation failed: {e}")
            return None
    
    def generate_submodular_explanations(self, 
                                       texts: List[str],
                                       sample_size: int = ExplanationConfig.DEFAULT_SAMPLE_SIZE,
                                       num_features: int = ExplanationConfig.DEFAULT_NUM_FEATURES,
                                       num_explanations: int = ExplanationConfig.DEFAULT_NUM_EXPLANATIONS) -> Any:
        """
        Generate submodular pick explanations for multiple instances.
        
        Args:
            texts: List of texts to explain
            sample_size: Sample size for submodular pick
            num_features: Number of features per explanation
            num_explanations: Number of explanations to generate
            
        Returns:
            SubmodularPick object with explanations
        """
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input must be a list of non-empty strings")
            
        try:
            sp_obj = submodular_pick.SubmodularPick(
                self.lime_explainer,
                texts,
                self.analyzer.predict_proba,
                sample_size=sample_size,
                num_features=num_features,
                num_exps_desired=num_explanations
            )
            return sp_obj
        except Exception as e:
            logging.error(f"Submodular pick explanation failed: {e}")
            raise
    
    def get_explanation_summary(self, 
                              text: str, 
                              index: int,
                              true_label: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a comprehensive summary of explanations for a text.
        
        Args:
            text: Text to explain
            index: Index of the text (for reference)
            true_label: True label if available
            
        Returns:
            Dictionary containing explanation summary
        """
        try:
            # Get predictions
            prediction = self.analyzer.predict_single(text)
            probabilities = self.analyzer.predict_proba([text])[0]
            
            # Get LIME explanation
            lime_explanation = self.explain_instance_lime(text)
            
            summary = {
                "document_id": index,
                "text": text,
                "prediction": prediction,
                "probabilities": {
                    "positive": float(probabilities[0]),
                    "negative": float(probabilities[1])
                },
                "true_label": true_label,
                "lime_explanation": lime_explanation.as_list() if lime_explanation else None,
                "lime_score": lime_explanation.score if lime_explanation else None
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Failed to generate explanation summary: {e}")
            raise
