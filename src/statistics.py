import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from difflib import get_close_matches

def get_class_distribution(data, label_column):
    """
    Calculate class distribution in the dataset.
    
    Args:
    data: Polars DataFrame
    label_column: Name of the label column
    
    Returns:
    Polars DataFrame with class distribution
    """
    total_count = len(data)
    return (
        data.group_by(label_column)
        .agg(pl.count().alias('counts'))
        .sort('counts', descending=True)
        .with_columns([
            (pl.col('counts') / total_count * 100).alias('percentage')
        ])
    )

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate classification metrics.
    
    Args:
    true_labels: List of true labels
    predicted_labels: List of predicted labels
    
    Returns:
    Dictionary of calculated metrics
    """
    return {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'precision': precision_score(true_labels, predicted_labels, average='weighted', zero_division=0),
        'recall': recall_score(true_labels, predicted_labels, average='weighted', zero_division=0),
        'f1': f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    }

def create_label_to_code_mapping(unique_labels, codebook):
    """
    Create a mapping between labels and codebook codes.
    
    Args:
    unique_labels: List of unique labels
    codebook: Dictionary containing the codebook or None
    
    Returns:
    Dictionary mapping labels to codebook codes or None if no codebook
    """
    if codebook is None:
        return None

    mapping = {}
    for label in unique_labels:
        label_number = ''.join(filter(str.isdigit, label))[:3]
        codebook_key = f"per{label_number}"
        
        if codebook_key in codebook:
            mapping[label] = codebook_key
        else:
            closest_matches = get_close_matches(codebook_key, codebook.keys(), n=1, cutoff=0.6)
            mapping[label] = closest_matches[0] if closest_matches else None
    
    return mapping