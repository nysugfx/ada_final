# Central Park Temperature Prediction Project
## Overview

This project aims to predict daily average temperatures in Central Park, NYC, using a combination of weather and environmental data. Through data collection, feature engineering, and machine learning models, the project seeks to provide accurate temperature forecasts.
## Files
- [`data/`](./data): Contains raw and processed datasets
- [`reports/figures/`](./reports/figures): Visualizations generated during analysis
- [`rf_reports/figures/`](./rf_reports/figures): Figures related to the Random Forest model
- [`KNN.ipynb`](./KNN.ipynb): Jupyter Notebook implementing the K-Nearest Neighbors model
- [`RF_model.ipynb`](./RF_model.ipynb): Jupyter Notebook for the Random Forest model
- [`gradient_boosting.py`](./gradient_boosting.py): Script implementing the Gradient Boosting model
- [`data_collection.py`](./data_collection.py): Script for aggregating and cleaning raw data
- [`eda.py`](./eda.py): Performs exploratory data analysis
- [`setup.py`](./setup.py): Project setup script
- [`pyproject.toml`](./pyproject.toml): Project metadata and dependency definitions
- [`features_readme.md`](./features_readme.md): Documentation of engineered features


## Models Used

- Random Forest Model
- Gradient Boosting Model
- K-Nearest Neighbers Model

Each model is trained and evaluated to determine its effectiveness in predicting temperature variations.

## Features Engineered

- Lag variables 
- Rolling means and standard deviations
- Seasonal and holiday indicators
- Interaction terms

## Collaborators
- [Jimin Park](jp4632)
- [Alexander Daniel Friedman](alex-friedman-modo)
- [Jiaheng Zhang](Chris010923)
- [Hanzhong Yang](Aspirine2212)
