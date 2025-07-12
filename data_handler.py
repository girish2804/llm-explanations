"""
Data loading and processing utilities.
"""
import datasets
from typing import List, Dict, Any, Optional
import logging

from config import DatasetConfig


class DataHandler:
    """Handles dataset loading and preprocessing."""
    
    def __init__(self, dataset_name: str = DatasetConfig.DATASET_NAME):
        """
        Initialize the data handler.
        
        Args:
            dataset_name: Name of the dataset to load
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the dataset from HuggingFace datasets."""
        try:
            self.dataset = datasets.load_dataset(self.dataset_name)
            logging.info(f"Successfully loaded dataset: {self.dataset_name}")
        except Exception as e:
            logging.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise
    
    def get_test_sample(self, index: int) -> Dict[str, Any]:
        """
        Get a specific test sample by index.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Dictionary containing the sample data
        """
        if not self.dataset or DatasetConfig.TEST_SPLIT not in self.dataset:
            raise ValueError("Dataset not loaded or test split not available")
            
        if index < 0 or index >= len(self.dataset[DatasetConfig.TEST_SPLIT]):
            raise IndexError(f"Index {index} out of range for test dataset")
            
        return self.dataset[DatasetConfig.TEST_SPLIT][index]
    
    def get_test_texts(self, start_index: int = 0, 
                      end_index: Optional[int] = None) -> List[str]:
        """
        Get a range of test texts.
        
        Args:
            start_index: Starting index (inclusive)
            end_index: Ending index (exclusive), if None uses default sample size
            
        Returns:
            List of text strings
        """
        if not self.dataset or DatasetConfig.TEST_SPLIT not in self.dataset:
            raise ValueError("Dataset not loaded or test split not available")
            
        if end_index is None:
            end_index = min(
                start_index + DatasetConfig.DEFAULT_SAMPLE_SIZE,
                len(self.dataset[DatasetConfig.TEST_SPLIT])
            )
        
        if start_index < 0 or end_index <= start_index:
            raise ValueError("Invalid index range")
            
        texts = []
        for i in range(start_index, end_index):
            sample = self.dataset[DatasetConfig.TEST_SPLIT][i]
            texts.append(sample['text'])
            
        return texts
    
    def get_test_labels(self, start_index: int = 0, 
                       end_index: Optional[int] = None) -> List[int]:
        """
        Get a range of test labels.
        
        Args:
            start_index: Starting index (inclusive)
            end_index: Ending index (exclusive)
            
        Returns:
            List of label integers
        """
        if not self.dataset or DatasetConfig.TEST_SPLIT not in self.dataset:
            raise ValueError("Dataset not loaded or test split not available")
            
        if end_index is None:
            end_index = min(
                start_index + DatasetConfig.DEFAULT_SAMPLE_SIZE,
                len(self.dataset[DatasetConfig.TEST_SPLIT])
            )
        
        if start_index < 0 or end_index <= start_index:
            raise ValueError("Invalid index range")
            
        labels = []
        for i in range(start_index, end_index):
            sample = self.dataset[DatasetConfig.TEST_SPLIT][i]
            labels.append(sample['label'])
            
        return labels
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        if not self.dataset:
            return {"status": "No dataset loaded"}
            
        info = {
            "dataset_name": self.dataset_name,
            "splits": list(self.dataset.keys()),
            "test_size": len(self.dataset[DatasetConfig.TEST_SPLIT]) if DatasetConfig.TEST_SPLIT in self.dataset else 0
        }
        
        return info
