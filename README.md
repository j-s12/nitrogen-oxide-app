# Nitrogen Oxide Level Prediction

A Flask-based machine learning web application that predicts nitrogen oxide levels using population density and industrial activity as input values.

## Features

- Predicts nitrogen oxide levels from user input
- Uses a trained machine learning model stored as a `.pkl` file
- Simple Flask web interface
- Form-based input and prediction display

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Pickle
- Machine Learning

## How It Works

1. The user enters population density and industrial activity values.
2. The Flask app sends the input to the trained model.
3. The model predicts the nitrogen oxide level.
4. The result is displayed on the web page.

## Project Structure

```text
project-folder/
│── app.py
│── nitrogen_oxide_model.pkl
│── templates/
│   └── index.html
│── README.md
