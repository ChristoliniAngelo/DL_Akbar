# Tomato Classification with Webcam

This Python application performs real-time tomato classification using a pre-trained machine learning model (.h5). The app captures frames from a webcam (internal or external) and classifies tomatoes as either ripe or unripe. It also displays the classification result and accuracy in the form of a bounding box on the detected tomato.

## Features

- Real-time tomato detection and classification.
- Ability to select between internal and external webcams.
- Visual display of predictions and classification accuracy.
- Termination of the process via a UI button or pressing `q`.

## Requirements

To install the required Python libraries, run:

pip install -r requirements.txt

## Usage
### Step 1: Prepare Your Model
Ensure that you have a pre-trained model saved in the .h5 format. Place this model in the root directory of the project with the name model.h5.

### Step 2: Run the Application
To start the application, simply run the Python script:
python app.py

### Step 3: Select Webcam
A UI will prompt you to select the webcam you'd like to use. Choose either the internal laptop webcam or an external one.

### Step 4: Real-Time Tomato Classification
Once the webcam is activated, the app will:
-Capture frames.
-Perform real-time classification using the provided model.
-Display bounding boxes around detected tomatoes with their predicted labels (ripe or unripe).
-Show classification accuracy in the form of confidence scores.
-Press q at any time to quit the application.

Let me know if you'd like any further adjustments!
