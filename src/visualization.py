import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_class_distribution(class_distribution, label_column):
    """
    Create a bar plot of class distribution.
    
    Args:
    class_distribution: Polars DataFrame with class distribution
    label_column: Name of the label column
    
    Returns:
    Plotly Figure
    """
    fig = px.bar(class_distribution.to_pandas(), x=label_column, y='counts', text='percentage',
                 labels={'counts': 'Count', label_column: 'Class'}, title="Class Distribution")
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig

def plot_confusion_matrix(cm, unique_labels):
    """
    Create a heatmap of the confusion matrix.
    
    Args:
    cm: Confusion matrix
    unique_labels: List of unique label values
    
    Returns:
    Plotly Figure
    """
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=unique_labels,
                    y=unique_labels,
                    color_continuous_scale='Viridis')
    fig.update_layout(title='Confusion Matrix')
    return fig

def display_multi_class_stats(metrics):
    """
    Display multi-class classification statistics.
    
    Args:
    metrics: Dictionary of calculated metrics
    """
    st.subheader("Coding Statistics")
    stats_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1 Score'],
        'Value': [f"{metrics['accuracy']:.2f}", f"{metrics['precision']:.2f}", 
                  f"{metrics['recall']:.2f}", f"{metrics['f1']:.2f}"]
    })
    st.table(stats_df)