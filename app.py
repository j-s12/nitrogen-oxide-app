from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open("nitrogen_oxide_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        population_density = float(request.form['population_density'])
        industrial_activity = float(request.form['industrial_activity'])

        # Make a prediction using the loaded model
        prediction = model.predict([[population_density, industrial_activity]])

        return render_template('index.html', prediction=f'Predicted Nitrogen Oxide Level: {prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
