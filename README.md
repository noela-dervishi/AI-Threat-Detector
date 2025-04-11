# Cybersecurity Threat Detection System

## Overview
The **Cybersecurity Threat Detection System** is a machine learning-based web application that detects network intrusions (attacks) in real-time using a trained **Random Forest** model on the NSL-KDD dataset. This system is designed to classify network traffic as either normal or an attack based on several features such as protocol type, service, source/destination bytes, and more.
The system uses **Streamlit** to deploy an interactive dashboard that allows users to input traffic data and receive predictions. It also provides visualizations and model evaluation metrics such as precision, recall, and F1-score.

## Description
The **Cybersecurity Threat Detection System** uses machine learning techniques to classify network traffic. The system is built using the **NSL-KDD dataset** and deployed as a **Streamlit** application. The machine learning model predicts whether a given network connection is "normal" or "an attack".

Key features include:
- **Real-time prediction**: Users can input traffic features and instantly get predictions (Normal/Attack).
- **Prediction History**: Displays a history of previous predictions and a count plot.
- **Model Evaluation**: Shows performance metrics like precision, recall, and confusion matrix.

### Components
1. **Model**: 
   - **RandomForestClassifier**: A robust machine learning model used to predict whether network traffic is normal or an attack.
2. **Preprocessing**: 
   - **Feature Scaling**: `MinMaxScaler` scales the numerical features to a range of [0, 1] to prepare them for model training.
   - **One-Hot Encoding**: Categorical features like `protocol_type`, `service`, and `flag` are one-hot encoded to be used in the model.
3. **Streamlit Dashboard**:
   - An interactive dashboard where users can input network features and receive real-time predictions.
   - Displays a history of past predictions with visualizations such as count plots and confusion matrices.
4. **Evaluation**:
   - **Confusion Matrix**: A visual representation of the model's performance, showing the count of true positives, true negatives, false positives, and false negatives.
   - **Classification Report**: Provides important metrics such as accuracy, precision, recall, and F1-score.

## Installation

### Prerequisites
1. **Python 3.7** or higher
2. Install the dependencies listed in `requirements.txt`.
### Install Dependencies
Create a virtual environment and install the required libraries using `pip`:
pip install -r requirements.txt
### Dataset
The model is trained on the **NSL-KDD dataset**. To train the model, you need to download the dataset (`KDDTrain+.csv`) from the UCI Machine Learning Repository or other sources. Place the dataset in the `data/` directory.

## Usage

### 1. Train the Model
To train the model, run the following command:
python main.py
This script performs the following tasks:
- Loads the **NSL-KDD dataset**.
- Preprocesses the data (one-hot encoding and feature scaling).
- Trains a **Random Forest classifier** on the data.
- Saves the trained model, scaler, and feature names to the `models/` directory.
After running this script, the model and preprocessing assets are ready for use in the dashboard.
### 2. Run the Streamlit Dashboard
Once the model is trained, you can start the **Streamlit dashboard** by running:
py -m streamlit run dashboard.py
This will launch a local server and open the dashboard in your browser. The dashboard allows users to:
- Input various network traffic features.
- Get predictions on whether the network traffic is "Normal" or an "Attack".
- View prediction history and distribution.
- View performance evaluation metrics (confusion matrix and classification report).

## Streamlit Dashboard Interface
The dashboard allows users to interact with the machine learning model via an intuitive interface.

### Input Features
The dashboard provides a form to input various network traffic features, such as:
- **Duration**: The duration of the connection (in seconds).
- **Protocol Type**: The protocol used for communication (e.g., TCP, UDP, ICMP).
- **Service**: The service being used for the communication (e.g., HTTP, FTP, SMTP).
- **Flag**: The flag associated with the connection (e.g., REJ, RSTO).
- **Source/Destination Bytes**: The amount of data transferred.
- **Login Information**: Various login-related features, such as the number of failed logins or whether the user is logged in.

### Prediction & History
- **Prediction**: Upon clicking the "Predict" button, the system outputs whether the network connection is "Normal" or an "Attack".
- **Prediction History**: Displays a history of past predictions, including a count plot of Normal vs. Attack predictions, and a table of the last 10 predictions.

### Model Evaluation
As users submit multiple predictions, the system computes and displays:
- **Confusion Matrix**: A heatmap that visualizes the performance of the model.
- **Classification Report**: A detailed report showing precision, recall, F1-score, and other evaluation metrics.

## Model Evaluation
The model performance is evaluated using metrics like:
- **Confusion Matrix**: Shows true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) for evaluating classification accuracy.
- **Precision**: Proportion of true positive predictions out of all positive predictions.
- **Recall**: Proportion of true positive predictions out of all actual positives.
- **F1-Score**: Harmonic mean of precision and recall, providing a single metric for model performance.

## Conclusion
The **Cybersecurity Threat Detection System** is a powerful tool for detecting network intrusions based on machine learning techniques. By utilizing a trained **Random Forest classifier** on the **NSL-KDD dataset**, this system offers real-time predictions of network traffic and can help in identifying malicious activities or attacks. The interactive Streamlit dashboard allows users to input network traffic data and receive instant feedback, while the model evaluation metrics help assess the performance and reliability of the model.

## Acknowledgments

- **NSL-KDD Dataset**: Used for training and evaluating the intrusion detection model.
- **Streamlit**: Open-source framework used for creating the interactive dashboard.
- **Scikit-learn**: Used for machine learning, model training, and evaluation.
