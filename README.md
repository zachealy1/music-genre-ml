# 🎶 Music Genre & Taste Classification with Machine Learning 🎶

This repository contains machine learning models for classifying music genres and analysing user tastes. These models can use data processing, feature extraction, and various machine learning techniques to identify genre categories and provide insights into personalised music preferences.

## Project Overview

This project aims to explore and analyse music data to predict music genres and model user music tastes. This repository includes two primary Jupyter notebooks:

- **ML-MusicGenre.ipynb**: A notebook focused on genre classification. This may involve extracting audio features and training models to classify songs by genre.
- **ML-MusicTaste2.ipynb**: A notebook dedicated to modelling music taste, potentially involving collaborative filtering, clustering, or other recommendation techniques to understand and predict user preferences.

## Contents

- `ML-MusicGenre.ipynb`: Implements a genre classification pipeline using machine learning models trained on music features.
- `ML-MusicTaste2.ipynb`: Develops a music recommendation or taste analysis model that identifies patterns in user listening data.
- `data/`: (Optional) Directory for storing raw music datasets and user preference data if applicable.
- `models/`: (Optional) Saved models and artefacts from training.

## Requirements

This project requires Python 3.7 or above and the following packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `librosa` (for audio feature extraction)
- `matplotlib` and `seaborn` (for visualisations)
- `tensorflow` or `torch` (if using deep learning models)

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Getting Started

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Run the Notebooks:** Open each notebook in Jupyter to explore or train the models:
```bash
jupyter notebook
```
- Start with `ML-MusicGenre.ipynb` to understand and execute the genre classification model.
- Proceed to `ML-MusicTaste2.ipynb` to analyse user taste and create personalised music recommendations.

## Project Workflow
**1. Data Preprocessing**
- Loading and cleaning music data (genres, user listening history).
- Feature extraction (e.g., spectral features, rhythm, timbre) using libraries like librosa.

**2. Model Training**
- Genre Classification: Use SVM, Random Forest, or Neural Networks classifiers.
- Taste Prediction: Build recommendation models like collaborative, clustering, or content-based filtering.

**3. Evaluation and Fine-Tuning**
- Evaluate model performance using metrics like accuracy, F1-score for classification, or RMSE for recommendations.
- Tune hyperparameters to improve performance.

**4. Visualisation**
- Display model performance with plots for accuracy, loss, confusion matrices, etc.
- Analyse user clusters or recommendations.

## Results
After training, the notebooks provide results such as:

- Classification accuracy for genre prediction.
- Personalised recommendations or taste analysis metrics.
