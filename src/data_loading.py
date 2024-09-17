import polars as pl
import streamlit as st

@st.cache_data
def load_data(file):
    """
    Load data from a file into a Polars DataFrame.
    
    Args:
    file: Uploaded file or default file path
    
    Returns:
    Polars DataFrame
    """
    try:
        if file is None:
            return pl.read_csv('data/preprocessed_data.csv')
        else:
            return pl.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None