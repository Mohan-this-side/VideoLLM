
# Multi-Modal Video Ad Classifier

This repository contains the implementation and report for a **Multi-Modal Video Ad Classifier**. The project aims to classify video advertisements by answering 21 binary yes/no questions using both textual and visual data from the ads. The classifier uses a combination of **Natural Language Processing (NLP)** for text-based features and **Convolutional Neural Networks (CNN)** for visual features. This approach allows the model to gain a comprehensive understanding of the content presented in the ads.

## Project Overview

The project involves the following key tasks:
- **Data Preprocessing**: Cleansing and standardizing textual data from ad descriptions and speech, tokenizing and normalizing the input.
- **Feature Extraction**:
  - **Textual Features**: Extracted using **DistilBERT** for transcriptions and descriptions.
  - **Visual Features**: Extracted from video frames using **ResNet3D**.
  - **Audio Features**: Extracted speech from video using **Whisper**.
  - **On-Screen Text Features**: Extracted using **EasyOCR**.
- **Model Architecture**: The classifier integrates textual and visual features, combining them using fully connected layers for final predictions. The architecture uses **AdamW optimizer** and **Binary Cross-Entropy Loss**.
- **Evaluation**: The classifier achieved an F1 score of **0.8088** and an accuracy of **0.8184** on the test dataset. The classifier performed exceptionally well for questions related to explicit features like product mentions and calls to action.

## Key Features
- **Multi-Modal Data Handling**: Combines text, audio, and visual inputs to provide robust video ad classification.
- **Feature Extraction Pipeline**: Automates the extraction of features from new video data using pre-trained models such as **ResNet3D**, **Whisper**, and **EasyOCR**.
- **Robust Performance**: The classifier achieves high accuracy in predicting explicit features but struggles with subjective questions, providing insights into areas of improvement.

## Files in This Repository

- **VideoLLM_Code.ipynb**: The main code file that contains the implementation of the multi-modal video ad classifier. The notebook includes functions for data preprocessing, feature extraction, model training, evaluation, and prediction.
- **VideoLLM_Report.pdf**: The project report that provides a detailed analysis of the methodology, results, and observations. The report discusses the challenges faced during classification and highlights the model’s performance on various questions.

## Getting Started

### Prerequisites

To run the code provided in `VideoLLM_Code.ipynb`, you will need the following:

- **Python 3.7+**
- **Jupyter Notebook** or **Google Colab**
- Required Python libraries:
  - PyTorch
  - Transformers (HuggingFace)
  - OpenCV
  - EasyOCR
  - Whisper

You can install these libraries using the following commands:

```bash
pip install torch torchvision transformers opencv-python easyocr
pip install git+https://github.com/openai/whisper.git
```

### Running the Code

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/VideoAdClassifier.git
   cd VideoAdClassifier
   ```

2. **Run the Jupyter Notebook**:
   Open the `VideoLLM_Code.ipynb` file using Jupyter or Google Colab and run the cells sequentially.

3. **Process New Videos**:
   To classify new video advertisements, use the `process_video()` function, which integrates feature extraction and classification. This function processes the video input and returns the predictions for the 21 binary questions.

### Example Usage

```python
# Example of processing a new video
video_path = 'path_to_your_video.mp4'
predictions = process_video(video_path)
print(predictions)
```

### Evaluation Metrics

The project uses **F1 Score**, **Precision**, **Recall**, and **Accuracy** to evaluate model performance. Below are the final metrics achieved:

- **F1 Score**: 0.8088
- **Precision**: 0.8115
- **Recall**: 0.8061
- **Accuracy**: 0.8184

## Results

The classifier performs well in recognizing objective features like product mentions and visual brand cues. However, it struggles with subjective aspects like emotional impact and creativity. In the future, improvements can be made to enhance the classifier’s ability to handle nuanced and complex ad content.

## Challenges

- Difficulty in interpreting complex marketing language.
- Low performance on subjective and visually-dependent questions.
- Zero F1 scores on questions related to subtle content (e.g., emotional impact, creative elements).

## Future Work

- **Improve Subjective Feature Classification**: Enhance the model’s ability to classify nuanced questions by refining feature extraction methods and providing more training data for subjective elements.
- **Visual Context Integration**: Improve the visual feature extraction to capture more meaningful data from video frames, addressing current blind spots in the classifier.
- **Data Augmentation**: Add more diverse datasets for training to cover a broader range of ad types and visual styles.

## Contact

For any questions or issues, feel free to reach out via email at [mohanbhosale6@gmail.com].

---
