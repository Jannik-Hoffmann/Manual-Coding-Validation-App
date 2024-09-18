import polars as pl
import streamlit as st
from pathlib import Path
import json

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
            current_dir = Path(__file__).parent.parent
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

@st.cache_data
def load_codebook(file):
    """
    Load codebook from a JSON file or use the default codebook.
    
    Args:
    file: Uploaded JSON file or None for default file
    
    Returns:
    Dictionary containing the codebook or None if no codebook is available
    """
    try:
        if file is None:
            current_dir = Path(__file__).parent
            default_file_path = current_dir / "data" / "default_codebook.json"
            st.info(f"Loading default codebook from {default_file_path}")
            if not default_file_path.exists():
                st.warning(f"Default codebook not found at {default_file_path}. No codebook will be used.")
                return None
            
            with open(default_file_path, 'r') as f:
                return json.load(f)
        else:
            with open(file, 'r') as file:
                return json.load(file)
    except Exception as e:
        st.error(f"Error loading codebook: {str(e)}")
        return None