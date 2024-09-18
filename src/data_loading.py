import polars as pl
import streamlit as st
from pathlib import Path

@st.cache_data
def load_data(file):
    """
    Load data from a file into a Polars DataFrame.
    
    Args:
    file: Uploaded file or None for default file
    
    Returns:
    Polars DataFrame
    """
    try:
        if file is None:
            # Get the directory of the current script
            current_dir = Path(__file__).parent.parent
            # Construct the path to the default dataset
            default_file_path = current_dir / "data" / "preprocessed_data.csv"
            
            if not default_file_path.exists():
                st.error(f"Default dataset not found at {default_file_path}. Please upload a CSV file.")
                return None
            
            return pl.read_csv(default_file_path)
        else:
            return pl.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None