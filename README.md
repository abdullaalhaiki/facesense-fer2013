# FaceSense — Facial Expression Recognition Using Deep Learning

## Project Overview

FaceSense is an interactive deep learning application that predicts facial-expression categories from uploaded photos or webcam captures.

The application uses a Convolutional Neural Network (CNN) developed with TensorFlow and Keras and deployed using Streamlit.

**Live Application:** https://facesense-fer2013-7mgjtwnu4cmqercb2pdtif.streamlit.app

## Dataset

**FER2013 — Facial Expression Recognition**

The dataset contains 35,887 grayscale facial images across seven classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The dataset is imbalanced, with significantly fewer Disgust examples.

## Data Preprocessing

1. Resized images to 48 × 48 pixels.
2. Converted images to grayscale.
3. Normalized pixel values to the range 0–1.
4. Split the original training data into training and validation subsets.
5. Preserved the separate 7,178-image test set for final evaluation.

## Model Architecture

Two architectures were explored.

**Baseline Dense Neural Network**

- Flatten layer
- Dense layer with 128 neurons
- Dropout
- Seven-class Softmax output

**Improved CNN**

- Three convolutional and max-pooling blocks
- ReLU activation
- Dense layer with 64 neurons
- Dropout for regularization
- Seven-class Softmax output
- Horizontal-flip data augmentation
- Adam optimizer
- Categorical cross-entropy loss

The CNN was trained for 12 epochs, with Early Stopping configured.

## Model Results

| Metric | Result |
|---|---|
| Baseline validation accuracy | 29.45% |
| CNN validation accuracy | 48.83% |
| CNN test accuracy | 48.61% |
| Macro-average F1 | 0.39 |
| Weighted-average F1 | 0.46 |

The CNN improved validation accuracy by **19.38 percentage points** compared with the baseline.

## Error Analysis

The confusion matrix revealed that the model performs relatively well on Happy and Surprise expressions but struggles with Fear and Disgust.

The model correctly identified no Disgust images in the test set. Sad was frequently confused with Neutral, while Fear was also confused with several other expressions.

These errors may result from class imbalance, ambiguous facial expressions, image quality, and limitations in the learned features.

## Streamlit Application

The application provides:

- Image upload
- Webcam photo capture
- Face detection using OpenCV
- Image preprocessing
- Expression classification
- Prediction scores for all seven classes
- A probability bar chart

## Running Locally

Use Python 3.11.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Technologies

Python, TensorFlow, Keras, NumPy, Pandas, OpenCV, Streamlit, Scikit-learn, and Matplotlib.

## Ethics and Limitations

The application predicts facial-expression labels, not a person's actual emotional or psychological state.

Results may be unreliable for unusual lighting, face angles, partial occlusions, and expressions underrepresented in the training dataset.

The model has limited overall accuracy and should not be used for medical, employment, security, or other consequential decisions. Webcam and uploaded images are processed to generate predictions; users should only submit images they have permission to use.

## Conclusion

This project demonstrated the full deep learning workflow, from dataset exploration and preprocessing to model development, evaluation, improvement, and cloud deployment.

Using a CNN improved classification accuracy compared with the Dense baseline. The final application demonstrates how a trained image classifier can be integrated into an interactive web application.

Further improvements could include additional training, class balancing, more data augmentation, and experimenting with deeper architectures.
