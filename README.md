
# Comprehensive Manual Coding Validation Tool

## Overview

The Comprehensive Manual Coding Validation Tool is a Streamlit-based web application designed to assist researchers in validating automated coding results through manual review and comparison. This tool provides an intuitive interface for reviewing coded data, adjusting labels, and analyzing the accuracy of automated coding processes.

## Live Demo

You can try out the tool without any installation by visiting our hosted Streamlit app:

[https://manual-coding-validation-tool.streamlit.app](https://manual-coding-validation-tool.streamlit.app)

## Features

- **Custom Dataset Upload**: Users can upload their own CSV datasets or use a default dataset for illustration.
- **Flexible Column Selection**: Select columns for text content, predicted labels, and additional information.
- **Multiple Sampling Methods**: Choose between Binary Classification, Multi-class Random Sampling, or Stratified Sampling.
- **Interactive Coding Interface**: Easily navigate through samples and adjust labels.
- **Real-time Statistics**: View accuracy, precision, recall, and F1 score updates as you code.
- **Confusion Matrix Visualization**: Understand classification performance with an interactive confusion matrix.
- **Progress Tracking**: Monitor your coding progress with a dynamic progress bar.
- **Export Options**: Save your validated samples in CSV, Excel, or JSON formats.
- **Dark Mode**: Toggle between light and dark themes for comfortable viewing.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/manual-coding-validation-tool.git
   ```

2. Navigate to the project directory:
   ```
   cd manual-coding-validation-tool
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`.

3. Follow the on-screen instructions to upload your dataset or use the default one.

4. Select the appropriate columns for text content and predicted labels.

5. Choose your sampling method and generate a sample.

6. Start coding! Review each item, adjust labels as needed, and submit.

7. Monitor your progress and view real-time statistics.

8. When finished, export your validated sample in your preferred format.

## Contributing

Contributions to improve the Comprehensive Manual Coding Validation Tool are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with a clear commit message.
4. Push your changes to your fork.
5. Submit a pull request with a description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your academic work, please cite it as follows:

```
[Your Name]. (2023). Comprehensive Manual Coding Validation Tool. [Computer software]. https://github.com/Jannik-Hoffmann/manual-classification-validation-tool
```

## Next Steps

I am working on improving and expanding the capabilities of this tool. Currently planned improvements:

1. **Collaborative Coding**: Enable multiple users to work on the same dataset simultaneously.
2. **Advanced Visualization**: Implement more complex visualizations for in-depth analysis of coding patterns.
3. **Multi-language Support**: Add support for multiple languages in the user interface.
4. **Mobile Responsiveness**: Optimize the interface for mobile devices to enable coding on-the-go.
5. **Performance Optimization**: Improve loading and processing times for larger datasets.

WeI welcome suggestions and contributions for these and other improvements!

## Contact

For questions, issues, or suggestions, please open an issue on the GitHub repository or contact [Your Name] at [your.email@example.com].
