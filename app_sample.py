import polars as pl
from typing import Dict

def get_class_distribution(data: pl.LazyFrame, label_column: str) -> Dict[str, int]:
    """
    Calculate the class distribution of the dataset.
    """
    distribution = (
        data.group_by(label_column)
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .collect()
    )
    return dict(zip(distribution[label_column], distribution['count']))

def get_stratified_sample(data: pl.LazyFrame, label_column: str, min_samples: int, max_samples: int) -> pl.DataFrame:
    """
    Generate a stratified sample from the dataset.
    """
    distribution = get_class_distribution(data, label_column)
    
    sample_sizes = {
        label: min(max(min_samples, count), max_samples)
        for label, count in distribution.items()
    }
    
    sampled_data = []
    for label, size in sample_sizes.items():
        class_data = (
            data.filter(pl.col(label_column) == label)
            .collect()
            .sample(n=size, shuffle=True)
        )
        sampled_data.append(class_data)
    
    return pl.concat(sampled_data)

# Parameters (adjust these as needed)
input_file = "C:/Users/janni/Desktop/GermaParlXML/updated_preprocessed_data.csv"
output_file = "stratified_sample.csv"
label_column = ""
min_samples_per_class = 2
max_samples_per_class = 5

# Load the data lazily
data = pl.scan_csv(input_file)

# Generate the stratified sample
sample = get_stratified_sample(data, label_column, min_samples_per_class, max_samples_per_class)

# Save the sample to a CSV file
sample.write_csv(output_file)

print(f"Stratified sample saved to {output_file}")

# Print sample statistics
sample_stats = sample.group_by(label_column).agg(pl.len().alias('count'))
print("\nSample class distribution:")
print(sample_stats)

total_samples = sample_stats['count'].sum()
print(f"\nTotal samples: {total_samples}")



input_file = "C:/Users/janni/Desktop/GermaParlXML/updated_preprocessed_data.csv"
output_file = "stratified_sample.csv"