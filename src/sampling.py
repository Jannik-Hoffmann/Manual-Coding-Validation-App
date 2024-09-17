import polars as pl
import math

def get_random_sample(data, sample_size):
    """
    Get a random sample from the dataset.
    
    Args:
    data: Polars DataFrame
    sample_size: Number of samples to take
    
    Returns:
    Polars DataFrame with random samples
    """
    return data.sample(n=min(sample_size, len(data)), seed=42)

def get_stratified_sample(data, label_column, min_samples_per_class, max_samples_per_class, class_distribution):
    """
    Get a stratified sample from the dataset.
    
    Args:
    data: Polars DataFrame
    label_column: Name of the label column
    min_samples_per_class: Minimum number of samples per class
    max_samples_per_class: Maximum number of samples per class
    class_distribution: Class distribution DataFrame
    
    Returns:
    Polars DataFrame with stratified samples
    """
    sample_sizes = {
        label: min(max(min_samples_per_class, count), max_samples_per_class)
        for label, count in zip(class_distribution['label'], class_distribution['counts'])
    }
    return data.sample(fraction=1.0, seed=42).over(label_column, sample_sizes)

def calculate_sample_size(confidence_level, margin_of_error, num_classes=2):
    """
    Calculate required sample size based on confidence level and margin of error.
    
    Args:
    confidence_level: Desired confidence level
    margin_of_error: Acceptable margin of error
    num_classes: Number of classes in the dataset
    
    Returns:
    Calculated sample size
    """
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    sample_size = (z_score**2 * 0.25 * num_classes) / (margin_of_error**2)
    return math.ceil(sample_size)