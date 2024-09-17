import streamlit as st
import polars as pl
import io
import base64

from src.data_loading import load_data
from src.sampling import get_random_sample, get_stratified_sample, calculate_sample_size
from src.statistics import get_class_distribution, calculate_metrics, get_confusion_matrix, suggest_sampling_method
from src.visualization import plot_class_distribution, plot_confusion_matrix, display_multi_class_stats

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Comprehensive Manual Coding Validation Tool")

# Define color schemes for light and dark modes
light_theme = {
    "bg_color": "#ffffff",
    "text_color": "#000000",
    "chart_color": "rgb(238, 238, 238)"
}
dark_theme = {
    "bg_color": "#0e1117",
    "text_color": "#ffffff",
    "chart_color": "rgb(17, 17, 17)"
}

# Function to toggle between light and dark mode
def toggle_theme():
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    st.session_state.dark_mode = not st.session_state.dark_mode

# Set the theme based on the current mode
theme = dark_theme if st.session_state.get('dark_mode', False) else light_theme

# Apply custom CSS for improved aesthetics
st.markdown(f"""
<style>
    .stButton>button {{ width: 100%; }}
    .stProgress .st-bo {{ background-color: #f63366; }}
    .stSelectbox {{ margin-bottom: 10px; }}
    .citation {{ background-color: {theme['chart_color']}; padding: 10px; border-radius: 5px; font-style: italic; }}
    body {{ color: {theme['text_color']}; background-color: {theme['bg_color']}; }}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Comprehensive Manual Coding Validation Tool")
    st.write("This tool helps researchers validate automated coding results through manual review and comparison.")

    # Dark mode toggle
    st.sidebar.button("Toggle Dark Mode", on_click=toggle_theme)

    # File uploader for custom dataset
    uploaded_file = st.file_uploader("Upload your own dataset (CSV)", type="csv")
    
    # Load the dataset
    full_data = load_data(uploaded_file)
    if full_data is None:
        st.stop()
    st.success(f"Dataset successfully loaded! Total population: {len(full_data):,} items")

    # Column selection
    st.subheader("Column Selection")
    text_column = st.selectbox("Select the column containing the text to be coded:", full_data.columns)
    # when choosing the label column, default to the first column with class in its name
    label_column = st.selectbox("Select the column containing the labels:", [col for col in full_data.columns if "class" in col.lower()])
    additional_columns = st.multiselect("Select additional columns to display (optional):", 
                                        [col for col in full_data.columns if col not in [text_column, label_column]])

    # Get number of unique classes and their distribution
    class_distribution = get_class_distribution(full_data, label_column)
    num_classes = len(class_distribution)
    st.write(f"Number of unique classes detected: {num_classes}")
    
    # Visualize class distribution
    st.subheader("Class Distribution")
    fig = plot_class_distribution(class_distribution, label_column)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Class distribution table:")
    st.write(class_distribution)

    # Suggest sampling method
    suggested_method = suggest_sampling_method(num_classes, class_distribution)
    st.write(f"Suggested sampling method: {suggested_method}")

    # Sampling method selection
    sampling_method = st.radio("Choose sampling method:", 
                               ["Binary Classification", "Multi-class Random Sampling", "Stratified Sampling"],
                               index=["binary", "multi-class", "stratified"].index(suggested_method))

    if sampling_method == "Binary Classification" or sampling_method == "Multi-class Random Sampling":
        confidence_level = st.selectbox("Select confidence level:", [0.95, 0.99], format_func=lambda x: f"{x*100}%")
        margin_of_error = st.slider("Select margin of error:", 0.01, 0.10, 0.05, 0.01, format="%0.2f")
        sample_size = calculate_sample_size(confidence_level, margin_of_error, num_classes if sampling_method == "Multi-class Random Sampling" else 2)
        st.write(f"Calculated sample size: {sample_size}")
        
        if st.button("Generate Sample"):
            st.session_state.coded_data = get_random_sample(full_data, sample_size)
            st.session_state.unique_labels = sorted(full_data[label_column].unique().to_list())
            st.session_state.current_index = 0
            st.session_state.manual_labels = []
            st.session_state.data_loaded = True
            st.rerun()

    elif sampling_method == "Stratified Sampling":
        min_samples_per_class = st.number_input("Minimum samples per class:", min_value=1, value=5)
        max_samples_per_class = st.number_input("Maximum samples per class:", min_value=min_samples_per_class, value=30)

        if st.button("Generate Stratified Sample"):
            st.session_state.coded_data = get_stratified_sample(full_data, label_column, min_samples_per_class, max_samples_per_class, class_distribution)
            st.session_state.unique_labels = sorted(full_data[label_column].unique().to_list())
            st.session_state.current_index = 0
            st.session_state.manual_labels = []
            st.session_state.data_loaded = True
            st.write(f"Stratified sample generated. Total samples: {len(st.session_state.coded_data)}")
            st.write("Sample class distribution:")
            sample_distribution = get_class_distribution(st.session_state.coded_data, label_column)
            st.write(sample_distribution)
            st.rerun()

    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        # Main coding interface
        st.subheader("Coding Interface")
        current_row = st.session_state.coded_data.row(st.session_state.current_index, named=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Text to Code ({st.session_state.current_index + 1}/{len(st.session_state.coded_data)}):**")
            st.write(current_row[text_column])
            if additional_columns:
                st.markdown("**Additional Information:**")
                for col in additional_columns:
                    st.write(f"{col}: {current_row[col]}")
        with col2:
            predicted_label = current_row[label_column]
            st.markdown(f"**Predicted Label:** {predicted_label}")
            
            default_index = st.session_state.unique_labels.index(predicted_label) if predicted_label in st.session_state.unique_labels else 0
            manual_label = st.selectbox("Select Manual Label", options=st.session_state.unique_labels, index=default_index)

        # Navigation and submission
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Previous") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
        with col2:
            if st.button("Submit"):
                st.session_state.manual_labels.append({
                    'text': current_row[text_column],
                    'predicted_label': predicted_label,
                    'manual_label': manual_label
                })
                
                if len(st.session_state.manual_labels) > 0:
                    true_labels = [item['manual_label'] for item in st.session_state.manual_labels]
                    predicted_labels = [item['predicted_label'] for item in st.session_state.manual_labels]
                    metrics = calculate_metrics(true_labels, predicted_labels)
                    display_multi_class_stats(metrics)
                    
                    cm = get_confusion_matrix(true_labels, predicted_labels, st.session_state.unique_labels)
                    fig = plot_confusion_matrix(cm, st.session_state.unique_labels)
                    st.plotly_chart(fig, use_container_width=True)

                if st.session_state.current_index < len(st.session_state.coded_data) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
                else:
                    st.success("Coding completed!")
        with col3:
            if st.button("Next ➡️") and st.session_state.current_index < len(st.session_state.coded_data) - 1:
                st.session_state.current_index += 1
                st.rerun()

        # Save results with export options
        if st.button("💾 Save Results"):
            results_df = pl.DataFrame(st.session_state.manual_labels)
            
            # Include additional columns if selected
            if additional_columns:
                for col in additional_columns:
                    results_df = results_df.with_columns(pl.Series(name=col, values=st.session_state.coded_data[col]))
            
            # Offer different export formats
            export_format = st.radio("Choose export format:", ["CSV", "Excel", "JSON"])
            
            if export_format == "CSV":
                csv = results_df.write_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="manually_coded_sample.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            elif export_format == "Excel":
                buffer = io.BytesIO()
                results_df.write_excel(buffer)
                b64 = base64.b64encode(buffer.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="manually_coded_sample.xlsx">Download Excel File</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:  # JSON
                json_str = results_df.write_json()
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="manually_coded_sample.json">Download JSON File</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            st.success("Results ready for download")

        # Progress bar
        st.progress(min((len(st.session_state.manual_labels) + 1) / len(st.session_state.coded_data), 1.0))

    # Citation information
    st.subheader("📚 How to Cite")
    st.markdown("""
    <div class="citation">
    If you use this tool in your academic work, please cite it as follows:
    
    Jannik Hoffmann. (2024).  Manual Coding Validation Tool. [Computer software]. https://github.com/Jannik-Hoffmann/manual-coding-validation-tools
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()