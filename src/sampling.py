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
    min_class_count = class_distribution['counts'].min()
    min_samples_per_class = min(min_samples_per_class, min_class_count)
    
    sample_sizes = {
        label: min(max(min_samples_per_class, count), max_samples_per_class)
        for label, count in zip(class_distribution[label_column], class_distribution['counts'])
    }
    
    sampled_data = []
    for label, size in sample_sizes.items():
        class_data = data.filter(pl.col(label_column) == label).sample(n=size, seed=42)
        sampled_data.append(class_data)
    
    return pl.concat(sampled_data)

def calculate_sample_size(confidence_level, margin_of_error, num_classes=2, expected_proportion=0.5):
    """
    Calculate required sample size based on confidence level, margin of error, and expected proportion.
    
    Args:
    confidence_level: Desired confidence level
    margin_of_error: Acceptable margin of error
    num_classes: Number of classes in the dataset
    expected_proportion: Expected proportion of the most prevalent class (default: 0.5)
    
    Returns:
    Calculated sample size
    """
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    sample_size = (z_score**2 * expected_proportion * (1 - expected_proportion) * num_classes) / (margin_of_error**2)
    return math.ceil(sample_size)