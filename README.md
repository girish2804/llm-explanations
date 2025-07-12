# Sentiment Analysis with Explanations

A modular implementation of sentiment analysis with LIME and SHAP explanations, including comprehensive fidelity testing.

## 📁 Project Structure

```
sentiment_analysis_explanations/
├── config.py                      # Configuration constants and enums
├── sentiment_analyzer.py          # Core sentiment analysis functionality
├── data_handler.py               # Data loading and processing
├── explanation_generator.py       # LIME and SHAP explanation generation
├── explanation_tester.py          # Fidelity testing framework
├── main_analysis.py              # Main analysis script
├── sentiment_analysis_explanation.ipynb  # Jupyter notebook
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

## 🛠️ Installation

1. Clone this repository or download the files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Using the Jupyter Notebook (Recommended)

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `sentiment_analysis_explanation.ipynb`
3. Run all cells to see the complete analysis

### Using the Python Script

```bash
python main_analysis.py
```

### Using Individual Modules

```python
from sentiment_analyzer import SentimentAnalyzer
from data_handler import DataHandler
from explanation_generator import ExplanationGenerator
from explanation_tester import ExplanationTester

# Initialize components
analyzer = SentimentAnalyzer()
data_handler = DataHandler()
explainer = ExplanationGenerator(analyzer)
tester = ExplanationTester(analyzer, explainer)

# Run analysis
sample_data = data_handler.get_test_sample(0)
explanation = explainer.explain_instance_lime(sample_data['text'])
```

## 🔧 Configuration

All configuration is centralized in `config.py`:

- **Model Settings**: Default model, device configuration
- **Dataset Settings**: Dataset name, split configurations
- **Explanation Settings**: Number of features, samples, etc.
- **Testing Settings**: Fidelity thresholds, test sizes

## 📈 Explanation Methods

### LIME (Local Interpretable Model-agnostic Explanations)
- Generates local explanations for individual predictions
- Identifies important words/features
- Provides confidence scores

### SHAP (SHapley Additive exPlanations)
- Provides global model explanations
- Based on game theory principles
- Unified framework for feature importance

## 🔍 Fidelity Testing

The system includes comprehensive fidelity testing:

### Supporting Fidelity
- Tests if emphasizing important features maintains predictions
- Measures explanation consistency

### Contrary Fidelity
- Tests if removing important features changes predictions
- Validates feature importance claims

### Metrics
- **Supporting Fidelity**: Proportion of cases where emphasizing important features maintains the original prediction
- **Contrary Fidelity**: Proportion of cases where removing important features changes the prediction
- **Average Fidelity**: Overall explanation quality metric
