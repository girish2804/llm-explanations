"""
Testing explanation fidelity and quality.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import random
from dataclasses import dataclass

from sentiment_analyzer import SentimentAnalyzer
from explanation_generator import ExplanationGenerator
from config import TestConfig


@dataclass
class FidelityResults:
    """Results from fidelity testing."""
    supporting_fidelity: float
    contrary_fidelity: float
    average_fidelity: float
    num_tests: int
    details: List[Dict[str, Any]]


class ExplanationTester:
    """Tests the fidelity and quality of explanations."""
    
    def __init__(self, analyzer: SentimentAnalyzer, explainer: ExplanationGenerator):
        """
        Initialize the explanation tester.
        
        Args:
            analyzer: Sentiment analyzer instance
            explainer: Explanation generator instance
        """
        self.analyzer = analyzer
        self.explainer = explainer
        random.seed(TestConfig.RANDOM_SEED)
        np.random.seed(TestConfig.RANDOM_SEED)
    
    def _modify_text_supporting(self, text: str, important_words: List[str]) -> str:
        """
        Modify text to support the explanation by emphasizing important words.
        
        Args:
            text: Original text
            important_words: List of important words from explanation
            
        Returns:
            Modified text with emphasized important words
        """
        words = text.split()
        modified_words = []
        
        for word in words:
            # Remove punctuation for comparison
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            
            if clean_word in [w.lower() for w in important_words]:
                # Emphasize by repetition (simple approach)
                modified_words.append(word + " " + word)
            else:
                modified_words.append(word)
        
        return " ".join(modified_words)
    
    def _modify_text_contrary(self, text: str, important_words: List[str]) -> str:
        """
        Modify text to contradict the explanation by removing important words.
        
        Args:
            text: Original text
            important_words: List of important words from explanation
            
        Returns:
            Modified text with important words removed or replaced
        """
        words = text.split()
        modified_words = []
        
        neutral_replacements = ["thing", "item", "aspect", "element", "part"]
        
        for word in words:
            # Remove punctuation for comparison
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            
            if clean_word in [w.lower() for w in important_words]:
                # Replace with neutral word or skip
                if random.random() < 0.5:
                    replacement = random.choice(neutral_replacements)
                    modified_words.append(replacement)
                # else: skip the word (removal)
            else:
                modified_words.append(word)
        
        return " ".join(modified_words)
    
    def test_explanation_fidelity(self, 
                                 text: str, 
                                 explanation: Any,
                                 num_top_features: int = 5) -> Dict[str, Any]:
        """
        Test the fidelity of an explanation.
        
        Args:
            text: Original text
            explanation: LIME explanation object
            num_top_features: Number of top features to test
            
        Returns:
            Dictionary with fidelity test results
        """
        try:
            # Get original prediction
            original_proba = self.analyzer.predict_proba([text])[0]
            original_prediction = np.argmax(original_proba)
            
            # Get important features from explanation
            explanation_list = explanation.as_list()
            top_features = [feature for feature, _ in explanation_list[:num_top_features]]
            
            # Test supporting fidelity
            supporting_text = self._modify_text_supporting(text, top_features)
            supporting_proba = self.analyzer.predict_proba([supporting_text])[0]
            supporting_prediction = np.argmax(supporting_proba)
            
            # Test contrary fidelity
            contrary_text = self._modify_text_contrary(text, top_features)
            contrary_proba = self.analyzer.predict_proba([contrary_text])[0]
            contrary_prediction = np.argmax(contrary_proba)
            
            # Calculate fidelity scores
            supporting_fidelity = 1.0 if supporting_prediction == original_prediction else 0.0
            contrary_fidelity = 1.0 if contrary_prediction != original_prediction else 0.0
            
            # Calculate probability differences
            supporting_prob_diff = abs(supporting_proba[original_prediction] - 
                                     original_proba[original_prediction])
            contrary_prob_diff = abs(contrary_proba[original_prediction] - 
                                   original_proba[original_prediction])
            
            return {
                "original_text": text,
                "original_prediction": int(original_prediction),
                "original_probabilities": original_proba.tolist(),
                "supporting_text": supporting_text,
                "supporting_prediction": int(supporting_prediction),
                "supporting_probabilities": supporting_proba.tolist(),
                "supporting_fidelity": supporting_fidelity,
                "supporting_prob_diff": float(supporting_prob_diff),
                "contrary_text": contrary_text,
                "contrary_prediction": int(contrary_prediction),
                "contrary_probabilities": contrary_proba.tolist(),
                "contrary_fidelity": contrary_fidelity,
                "contrary_prob_diff": float(contrary_prob_diff),
                "top_features": top_features,
                "explanation_scores": explanation_list
            }
            
        except Exception as e:
            logging.error(f"Fidelity test failed: {e}")
            raise
    
    def test_batch_fidelity(self, 
                           texts: List[str], 
                           sample_size: int = TestConfig.DEFAULT_TEST_SAMPLE_SIZE) -> FidelityResults:
        """
        Test fidelity for a batch of texts.
        
        Args:
            texts: List of texts to test
            sample_size: Number of texts to test (randomly sampled)
            
        Returns:
            FidelityResults object with aggregated results
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
            
        # Sample texts if needed
        if len(texts) > sample_size:
            texts = random.sample(texts, sample_size)
        
        supporting_scores = []
        contrary_scores = []
        test_details = []
        
        for i, text in enumerate(texts):
            try:
                # Generate explanation
                explanation = self.explainer.explain_instance_lime(text)
                
                # Test fidelity
                fidelity_result = self.test_explanation_fidelity(text, explanation)
                
                supporting_scores.append(fidelity_result["supporting_fidelity"])
                contrary_scores.append(fidelity_result["contrary_fidelity"])
                test_details.append(fidelity_result)
                
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i + 1}/{len(texts)} fidelity tests")
                    
            except Exception as e:
                logging.warning(f"Failed to test fidelity for text {i}: {e}")
                continue
        
        # Calculate aggregate scores
        supporting_fidelity = np.mean(supporting_scores) if supporting_scores else 0.0
        contrary_fidelity = np.mean(contrary_scores) if contrary_scores else 0.0
        average_fidelity = (supporting_fidelity + contrary_fidelity) / 2.0
        
        return FidelityResults(
            supporting_fidelity=supporting_fidelity,
            contrary_fidelity=contrary_fidelity,
            average_fidelity=average_fidelity,
            num_tests=len(test_details),
            details=test_details
        )
    
    def generate_fidelity_report(self, fidelity_results: FidelityResults) -> str:
        """
        Generate a human-readable fidelity report.
        
        Args:
            fidelity_results: Results from fidelity testing
            
        Returns:
            Formatted report string
        """
        report = f"""
Explanation Fidelity Report
==========================

Summary:
- Tests conducted: {fidelity_results.num_tests}
- Supporting fidelity: {fidelity_results.supporting_fidelity:.3f}
- Contrary fidelity: {fidelity_results.contrary_fidelity:.3f}
- Average fidelity: {fidelity_results.average_fidelity:.3f}

Interpretation:
- Supporting fidelity measures how often emphasizing important features
  maintains the original prediction (higher is better)
- Contrary fidelity measures how often removing important features
  changes the original prediction (higher is better)
- Average fidelity is the overall explanation quality metric

Threshold Analysis:
- Fidelity above {TestConfig.FIDELITY_THRESHOLD}: {'PASS' if fidelity_results.average_fidelity >= TestConfig.FIDELITY_THRESHOLD else 'FAIL'}
"""
        
        return report
