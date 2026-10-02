# Solar Power Forecasting Using Machine Learning and Apache Kafka

## 1. Project Overview

This project develops a machine learning based solar power forecasting system that predicts AC power generation using historical solar plant data.

The system combines:

- Data preprocessing
- Feature engineering
- Random Forest regression
- Time-series model evaluation
- Apache Kafka for real-time data streaming
- Real-time ML prediction
- Kafka-based forecast output

The final system can receive solar measurements through Kafka, process them using the trained Random Forest model, and publish the predicted AC power to another Kafka topic.

---

## 2. Problem Statement

Solar power generation varies depending on environmental and temporal conditions such as:

- Solar irradiation
- Ambient temperature
- Module temperature
- Time of day
- Historical power generation

Accurate forecasting can help with solar plant monitoring, energy planning, and operational decision-making.

The objective of this project is to build a machine learning model that predicts solar AC power and integrate the model with Apache Kafka for real-time forecasting.

---

## 3. Objectives

1. Prepare and clean solar power generation data.
2. Create useful time and historical power features.
3. Train a machine learning regression model.
4. Evaluate the model using MAE, RMSE and R².
5. Deploy the trained model into a Kafka-based streaming pipeline.
6. Generate real-time solar power forecasts.
7. Publish forecast results through Kafka.

---

## 4. Technologies Used

### Programming

- Python

### Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

### Machine Learning Model

- Random Forest Regressor

### Streaming

- Apache Kafka 4.3.1
- kafka-python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib

### Development Environment

- Windows
- Python Virtual Environment
- PowerShell

---

## 5. Dataset

The project uses solar power generation data containing variables such as:

- DATE_TIME
- AC_POWER
- DC_POWER
- AMBIENT_TEMPERATURE
- MODULE_TEMPERATURE
- IRRADIATION

Time-based features and historical AC power features were created for forecasting.

---

## 6. Final Model Features

The final Random Forest model uses the following 9 features:

1. IRRADIATION
2. AMBIENT_TEMPERATURE
3. MODULE_TEMPERATURE
4. HOUR
5. MINUTE
6. DAY_OF_YEAR
7. AC_POWER_LAG_1
8. AC_POWER_LAG_2
9. AC_POWER_LAG_3

The lag features provide information about recent power generation.

---

## 7. Machine Learning Workflow

```text
Raw Solar Dataset
       |
       v
Data Cleaning
       |
       v
Feature Engineering
       |
       v
Time-Series Dataset
       |
       v
Train/Test Split
       |
       v
Random Forest Regression
       |
       v
Model Evaluation
       |
       v
Final Model