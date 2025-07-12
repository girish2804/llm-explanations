"""
Main analysis script for sentiment analysis with explanations.
This script demonstrates the refactored system with proper modularity.
"""
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import our modules
from config import ExplanationConfig, DatasetConfig
from sentiment_analyzer import SentimentAnalyzer
from data_handler import DataHandler
from explanation_generator import ExplanationGenerator
from explanation_tester import ExplanationTester


def install_dependencies():
    """Install required dependencies."""
    import subprocess
    import sys
    
    dependencies = [
        'lime',
        'datasets',
        'transformers',
        'accelerate',
        'bitsandbytes',
        'sentencepiece',
        'shap',
        'matplotlib',
        'seaborn'
    ]
    
    for package in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            logging.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            logging.warning(f"Failed to install {package}")


def demonstrate_basic_analysis():
    """Demonstrate basic sentiment analysis functionality."""
    logging.info("=== Basic Sentiment Analysis Demo ===")
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    data_handler = DataHandler()
    
    # Display system information
    print(f"Device: {analyzer.get_device_info()}")
    print(f"Dataset info: {data_handler.get_dataset_info()}")
    
    # Test on sample data
    sample_texts = data_handler.get_test_texts(0, 3)
    
    print("\nSample texts and predictions:")
    for i, text in enumerate(sample_texts):
        prediction = analyzer.predict_single(text)
        probabilities = analyzer.predict_proba([text])[0]
        
        print(f"\nText {i}: {text[:100]}...")
        print(f"Prediction: {prediction}")
        print(f"Probabilities [positive, negative]: {probabilities}")


def demonstrate_lime_explanations():
    """Demonstrate LIME explanation generation."""
    logging.info("=== LIME Explanations Demo ===")
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    data_handler = DataHandler()
    explainer = ExplanationGenerator(analyzer)
    
    # Get a sample for explanation
    sample_index = 22
    sample_data = data_handler.get_test_sample(sample_index)
    
    print(f"\nGenerating explanation for document {sample_index}")
    print(f"Text: {sample_data['text'][:200]}...")
    print(f"True label: {sample_data['label']}")
    
    # Generate explanation summary
    explanation_summary = explainer.get_explanation_summary(
        sample_data['text'], 
        sample_index, 
        sample_data['label']
    )
    
    print(f"\nPrediction: {explanation_summary['prediction']}")
    print(f"Probabilities: {explanation_summary['probabilities']}")
    print(f"LIME explanation features: {explanation_summary['lime_explanation']}")
    
    # Display explanation in notebook format (if available)
    try:
        lime_explanation = explainer.explain_instance_lime(sample_data['text'])
        lime_explanation.show_in_notebook(text=True)
    except Exception as e:
        logging.warning(f"Could not display notebook explanation: {e}")


def demonstrate_shap_explanations():
    """Demonstrate SHAP explanation generation."""
    logging.info("=== SHAP Explanations Demo ===")
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    data_handler = DataHandler()
    explainer = ExplanationGenerator(analyzer)
    
    # Get background texts for SHAP
    background_texts = data_handler.get_test_texts(0, 20)
    
    # Get a sample for explanation
    sample_index = 22
    sample_data = data_handler.get_test_sample(sample_index)
    
    print(f"\nGenerating SHAP explanation for document {sample_index}")
    print(f"Text: {sample_data['text'][:200]}...")
    
    # Generate SHAP explanation
    shap_explanation = explainer.explain_instance_shap(
        sample_data['text'], 
        background_texts
    )
    
    if shap_explanation is not None:
        print("SHAP explanation generated successfully")
        # Additional SHAP visualization could be added here
    else:
        print("SHAP explanation failed or not available")


def demonstrate_submodular_explanations():
    """Demonstrate submodular pick explanations."""
    logging.info("=== Submodular Pick Explanations Demo ===")
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    data_handler = DataHandler()
    explainer = ExplanationGenerator(analyzer)
    
    # Get sample texts
    sample_texts = data_handler.get_test_texts(0, 10)
    
    print(f"\nGenerating submodular explanations for {len(sample_texts)} texts")
    
    # Generate submodular explanations
    sp_obj = explainer.generate_submodular_explanations(
        sample_texts,
        sample_size=3,
        num_features=6,
        num_explanations=2
    )
    
    print(f"Generated {len(sp_obj.sp_explanations)} submodular explanations")
    
    # Display explanation plots
    try:
        figures = [exp.as_pyplot_figure() for exp in sp_obj.sp_explanations]
        print("Explanation plots generated successfully")
        
        # Show plots with labels
        labeled_figures = [
            exp.as_pyplot_figure(label=exp.available_labels()[0]) 
            for exp in sp_obj.sp_explanations
        ]
        print("Labeled explanation plots generated successfully")
        
    except Exception as e:
        logging.warning(f"Could not generate explanation plots: {e}")


def demonstrate_fidelity_testing():
    """Demonstrate explanation fidelity testing."""
    logging.info("=== Fidelity Testing Demo ===")
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    data_handler = DataHandler()
    explainer = ExplanationGenerator(analyzer)
    tester = ExplanationTester(analyzer, explainer)
    
    # Get test texts
    test_texts = data_handler.get_test_texts(0, 20)
    
    print(f"\nTesting explanation fidelity on {len(test_texts)} samples")
    
    # Run fidelity tests
    fidelity_results = tester.test_batch_fidelity(test_texts, sample_size=10)
    
    # Generate and display report
    report = tester.generate_fidelity_report(fidelity_results)
    print(report)
    
    # Display detailed results for first few tests
    print("\nDetailed results for first 3 tests:")
    for i, detail in enumerate(fidelity_results.details[:3]):
        print(f"\nTest {i+1}:")
        print(f"  Original prediction: {detail['original_prediction']}")
        print(f"  Supporting fidelity: {detail['supporting_fidelity']}")
        print(f"  Contrary fidelity: {detail['contrary_fidelity']}")
        print(f"  Top features: {detail['top_features']}")


def run_comprehensive_analysis():
    """Run a comprehensive analysis combining all features."""
    logging.info("=== Comprehensive Analysis ===")
    
    print("Running comprehensive sentiment analysis with explanations...")
    
    try:
        # Install dependencies
        install_dependencies()
        
        # Run all demonstrations
        demonstrate_basic_analysis()
        demonstrate_lime_explanations()
        demonstrate_shap_explanations()
        demonstrate_submodular_explanations()
        demonstrate_fidelity_testing()
        
        print("\n=== Analysis Complete ===")
        print("All components have been successfully demonstrated.")
        
    except Exception as e:
        logging.error(f"Comprehensive analysis failed: {e}")
        raise


if __name__ == "__main__":
    run_comprehensive_analysis()