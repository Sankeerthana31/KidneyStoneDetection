# Deep Learning Framework for Automated Kidney Stone Localization and Classification in CT Images

## Project Overview

This project uses deep learning techniques to classify CT images as Positive or Negative and localize kidney stone regions using image segmentation.

The project integrates classification and stone localization models into a Streamlit web application.

## Features

- CT image classification using ClfNet.
- Kidney stone region localization using Stone U-Net.
- Predicted segmentation mask visualization.
- Stone-region overlay and bounding-box visualization.
- Enlarged view of the detected region.
- Interactive Streamlit application.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- Matplotlib
- TIFFFile
- Streamlit

## Project Structure

```text
KidneyStoneDetection/
|-- app.py
|-- requirements.txt
|-- README.md
|-- models/
|   |-- clfnet.py
|   |-- losses.py
|   |-- saved/
|   |   |-- clfnet_fold1_best.keras
|   |-- localization/
|       |-- stone_unet_512_best.keras
|-- preprocessing/
|-- prediction/
|-- training/
|-- evaluation/
|-- visualization/
```

## Evaluation Results

### Classification

- Five-fold cross-validation accuracy: 79.17%
- Holdout test accuracy: 79.64%

### Stone Localization

- Test images: 127
- Mean Dice score: 93.81%
- Median Dice score: 97.78%

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Sankeerthana31/KidneyStoneDetection.git
cd KidneyStoneDetection
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## Input Requirements

- Classification: JPG, JPEG, or PNG CT images.
- Stone localization: Compatible 512 × 512 grayscale TIFF images.

## Limitations

- This is an academic prototype and has not been clinically validated.
- Classification and segmentation are evaluated separately.
- The system identifies stone regions; it does not classify stone composition.

## Future Scope

- Improve classification accuracy.
- Integrate classification and localization into a jointly evaluated pipeline.
- Validate the system using larger and more diverse datasets.

## Author

**Sankeerthana Bompally**

GitHub: https://github.com/Sankeerthana31
