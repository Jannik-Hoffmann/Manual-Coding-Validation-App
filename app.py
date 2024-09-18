import streamlit as st
import polars as pl
import pandas as pd
import io
import base64
import json

from src.data_loading import load_data,load_codebook
from src.sampling import get_random_sample, get_stratified_sample, calculate_sample_size
from src.statistics import get_class_distribution, calculate_metrics, get_confusion_matrix, suggest_sampling_method, create_label_to_code_mapping
from src.visualization import plot_class_distribution, plot_confusion_matrix, display_multi_class_stats

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Comprehensive Manual Coding Validation Tool")

# Apply custom CSS for improved aesthetics
st.markdown("""
<style>
    .stButton>button { width: 100%; }
    .stProgress .st-bo { background-color: #f63366; }
    .stSelectbox { margin-bottom: 10px; }
    .citation { background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-style: italic; }
    .explanation { background-color: #e1f5fe; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)
def display_codebook(codebook):
    st.subheader("Complete Codebook")
    df = pd.DataFrame([
        {
            "Code": code,
            "Name": details['name'],
            "Description": details['description'],
            "Domain": details['domain']
        }
        for code, details in codebook.items()
    ])
    st.dataframe(df, height=400)  # Adjust height as needed

def main():
    st.title("Comprehensive Manual Coding Validation Tool")
    
    st.markdown("""
    <div class="explanation">
    <h3>Welcome to the Comprehensive Manual Coding Validation Tool</h3>
    <p>This tool helps researchers validate automated coding results through manual review and comparison. It provides a streamlined workflow for:</p>
    <ul>
        <li>Loading and preprocessing your dataset</li>
        <li>Uploading and integrating your codebook</li>
        <li>Selecting appropriate sampling methods</li>
        <li>Conducting manual coding validation</li>
        <li>Calculating and visualizing performance metrics</li>
    </ul>
    <p>By using this tool, you can ensure the accuracy and reliability of your automated coding processes, 
    leading to more robust research outcomes. Follow the steps below to get started with your validation process.</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploader for custom dataset
    uploaded_file = st.file_uploader("Upload your own dataset (CSV)", type="csv")
    
    # Load the dataset
    full_data = load_data(uploaded_file)
    if full_data is None:
        st.stop()
    st.success(f"Dataset successfully loaded! Total population: {len(full_data):,} items")

    # Codebook section
    st.subheader("Codebook")
    st.markdown("""
    <div class="explanation">
    <h4>What is a codebook and why is it useful?</h4>
    <p>A codebook is a document that provides detailed information about the codes or labels used in your dataset. It typically includes:</p>
    <ul>
        <li>Code identifiers</li>
        <li>Code names or short descriptions</li>
        <li>Detailed explanations of what each code means</li>
        <li>Examples or guidelines for when to apply each code</li>
    </ul>
    <p>Using a codebook can:</p>
    <ul>
        <li>Enhance consistency in manual coding</li>
        <li>Provide quick reference during the coding process</li>
        <li>Improve the reliability and validity of your coding results</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    use_codebook = st.checkbox("Use a codebook for this project", value=True)

    if use_codebook:
        st.markdown("""
        <div class="explanation">
        <h4>How to create a codebook JSON file:</h4>
        <p>Your codebook should be in JSON format with the following structure:</p>
        <pre>
        {
            "code_identifier": {
                "name": "Short name or title",
                "description": "Detailed explanation of the code",
                "domain": "Category or domain of the code (optional)"
            },
            ...
        }
        </pre>
        <p>To easily create this file:</p>
        <ol>
            <li>Copy your existing codebook (e.g., from a PDF or document)</li>
            <li>Use an AI language model (like ChatGPT) with the following prompt:</li>
        </ol>
        <pre>
        I have a codebook for my research project. Please convert it into a JSON format with the following structure for each code:
        {
            "code_identifier": {
                "name": "Short name or title",
                "description": "Detailed explanation of the code",
                "domain": "Category or domain of the code (optional)"
            }
        }
        Here's my codebook content:
        [Paste your codebook content here]
        </pre>
        <p>Review and adjust the AI-generated JSON as needed before using it in this tool.</p>
        </div>
        """, unsafe_allow_html=True)

        codebook_file = st.file_uploader("Upload Custom Codebook JSON (optional)", type="json")
        
        # Load the codebook (custom if uploaded, otherwise default)
        codebook = load_codebook(codebook_file)
        if codebook is None:
            st.warning("No codebook available. Please check if the default codebook is present or upload a custom codebook.")
        else:
            st.success("Codebook successfully loaded!")

            # Add a section to display the full codebook
            if st.button("View Complete Codebook", key="view_codebook"):
                display_codebook(codebook)
    else:
        codebook = None
        st.info("No codebook will be used for this session.")

    # Column selection
    st.subheader("Column Selection")
    text_column = st.selectbox("Select the column containing the text to be coded:", full_data.columns)
    remaining_columns = [col for col in full_data.columns if col != text_column]
    label_column = st.selectbox("Select the column containing the predicted labels:", remaining_columns)
    additional_columns = st.multiselect("Select additional columns to display (optional):", 
                                        [col for col in remaining_columns if col != label_column])

    # Confirm column selection
    if st.button("Confirm Column Selection"):
        st.session_state.columns_confirmed = True

    # Only proceed with analysis if columns are confirmed
    if 'columns_confirmed' in st.session_state and st.session_state.columns_confirmed:
        # Get number of unique classes and their distribution
        class_distribution = get_class_distribution(full_data, label_column)
        num_classes = len(class_distribution)
        st.write(f"Number of unique classes detected: {num_classes}")
        
        # Create label to code mapping
        unique_labels = full_data[label_column].unique().to_list()
        label_to_code_mapping = create_label_to_code_mapping(unique_labels, codebook)
        
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
        st.markdown("""
        <div class="explanation">
        <h3>Sampling Method and Size</h3>
        <p>Choosing the right sampling method and size is crucial for ensuring the validity and reliability of your manual coding validation process. Here's what you need to know:</p>
        <ul>
            <li><strong>Binary Classification:</strong> Used when you have only two classes. It's simple and straightforward, but may not be suitable for complex multi-class problems.</li>
            <li><strong>Multi-class Random Sampling:</strong> Suitable for datasets with multiple classes where each class is well-represented. It ensures each item has an equal chance of being selected.</li>
            <li><strong>Stratified Sampling:</strong> Ideal for datasets with imbalanced class distributions. It ensures that the sample maintains the same class proportions as the full dataset.</li>
        </ul>
        <p>The sample size is calculated based on your desired confidence level and margin of error. A larger sample size increases precision but requires more manual coding effort.</p>
        </div>
        """, unsafe_allow_html=True)

        sampling_method = st.radio("Choose sampling method:", 
                                   ["Binary Classification", "Multi-class Random Sampling", "Stratified Sampling"],
                                   index=["binary", "multi-class", "stratified"].index(suggested_method))

        if sampling_method == "Binary Classification" or sampling_method == "Multi-class Random Sampling":
            confidence_level = st.selectbox("Select confidence level:", [0.95, 0.99], format_func=lambda x: f"{x*100}%",
                                            help="Choose the desired confidence level for your sample.")
            margin_of_error = st.slider("Select margin of error:", 0.01, 0.10, 0.05, 0.01, format="%0.2f",
                                        help="Choose the acceptable margin of error for your sample.")
            expected_proportion = st.slider("Expected proportion:", 0.1, 0.9, 0.5, 0.1,
                                            help="Estimate the expected proportion of the most prevalent class. Use 0.5 if unsure.")
            sample_size = calculate_sample_size(confidence_level, margin_of_error, num_classes if sampling_method == "Multi-class Random Sampling" else 2, expected_proportion)
            st.write(f"Calculated sample size: {sample_size}")
            
            if st.button("Generate Sample"):
                st.session_state.coded_data = get_random_sample(full_data, sample_size)
                st.session_state.unique_labels = sorted(full_data[label_column].unique().to_list())
                st.session_state.current_index = 0
                st.session_state.manual_labels = []
                st.session_state.data_loaded = True
                st.rerun()

        elif sampling_method == "Stratified Sampling":
            min_samples_per_class = st.number_input("Minimum samples per class:", min_value=1, value=5,
                                                    help="Set the minimum number of samples to include for each class.")
            max_samples_per_class = st.number_input("Maximum samples per class:", min_value=min_samples_per_class, value=30,
                                                    help="Set the maximum number of samples to include for each class.")

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

        # Add a section to display the full codebook
        if st.button("View Complete Codebook"):
            display_codebook(codebook)

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
            
            # Display coding instructions if available and codebook is used
            if codebook and label_to_code_mapping:
                matching_code = label_to_code_mapping.get(manual_label)
                if matching_code and matching_code in codebook:
                    st.markdown("**Coding Instructions:**")
                    st.markdown(f"**Domain:** {codebook[matching_code]['domain']}")
                    st.markdown(f"**Code:** {matching_code}")
                    st.markdown(f"**{codebook[matching_code]['name']}**")
                    st.markdown(codebook[matching_code]['description'])
                else:
                    st.warning("No matching coding instructions found for this label.")

        # Add toggle for calculating statistics
        calculate_stats = st.toggle("Calculate and display statistics", value=False)

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
            # Create a DataFrame with all columns from the original sample
            results_df = st.session_state.coded_data.clone()
            
            # Create a dictionary to map text to manual labels
            manual_label_dict = {item['text']: item['manual_label'] for item in st.session_state.manual_labels}
            
            # Add manual labels and comparison column
            results_df = results_df.with_columns([
                pl.col(text_column).map_dict(manual_label_dict).alias('manual_label'),
                (pl.col(label_column) == pl.col(text_column).map_dict(manual_label_dict)).alias('labels_match')
            ])
            
            # Replace None values in manual_label column with original label
            results_df = results_df.with_columns([
                pl.when(pl.col('manual_label').is_null()).then(pl.col(label_column)).otherwise(pl.col('manual_label')).alias('manual_label')
            ])
            
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
        progress = min((len(st.session_state.manual_labels) + 1) / len(st.session_state.coded_data), 1.0)
        st.progress(progress)
        st.write(f"Progress: {progress:.1%}")

    # Citation information
    st.subheader("📚 How to Cite")
    st.markdown("""
    <div class="citation">
    If you use this tool in your academic work, please cite it as follows:
    
    Jannik Hoffmann. (2024). Manual Coding Validation Tool. [Computer software]. https://github.com/Jannik-Hoffmann/manual-coding-validation-tools
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()  