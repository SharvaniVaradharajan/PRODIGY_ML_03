# Image Classification using Support Vector Machine (SVM)

This project involves training a Support Vector Machine (SVM) classifier to classify images of cats and dogs from the Kaggle dataset.

## Project Overview

The goal is to build a machine learning model that can accurately distinguish between images of cats and dogs. SVM is chosen as the classification algorithm due to its effectiveness in handling high-dimensional data and its ability to find optimal decision boundaries.

## Dataset

- `train`: Directory containing the training images of cats and dogs.
- `test`: Directory containing the test images of cats and dogs.

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- keras
- tensorflow
## Explanation of the Code

### Data Preparation:

- The script uses `ImageDataGenerator` to load and preprocess images from the training directory.
- Images are rescaled by a factor of `1./255` and split into training and validation sets.
- `train_generator` and `validation_generator` are used to fetch the training and validation images respectively.

### Flattening the Images:

- The images are flattened from a 3D array (64x64x3) to a 1D array (64*64*3) so that they can be used as input to the SVM classifier.

### Model Training:

- An SVM model with a linear kernel is created using `make_pipeline` with `StandardScaler` to normalize the data.
- The model is trained on the flattened training images (`X_train`) and their corresponding labels (`y_train`).
### Evaluation

- The accuracy of the SVM model on the validation set is computed using `best_pipeline.score`.
- Predictions on the validation set are made using `best_pipeline.predict`.

### Classification Report

- A detailed classification report is generated using `classification_report`, which includes precision, recall, F1-score, and support for each class.
- The classification report is saved to a text file (`classification_report.txt`).

### Confusion Matrix

- A confusion matrix is computed using `confusion_matrix` to visualize the performance of the classifier.
- The confusion matrix is plotted using `seaborn.heatmap` and saved as an image (`confusion_matrix.png`).
## Results

The performance of the SVM classifier is assessed using the evaluation metrics mentioned above. The results include:

- **Accuracy**: The overall accuracy of the model on the test dataset.
- **Classification Report**: Detailed metrics for each class.
- **Confusion Matrix**: Visual representation of the classifier's performance.

## Acknowledgments

The dataset used in this project is provided by Kaggle.
