from flask import Flask, render_template, request, url_for, redirect, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

filename = "model/model_pickle.pkl"
model = pickle.load(open(filename, "rb"))

cols = ['Age', 'Potential', 'Value', 'Special', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Dribbling', 'Curve', 'FKAccuracy',
       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
       'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
       'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision',
       'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',
       'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

# Set up the main route
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    features = [int (i) for i in request.form.values()]
    input_features = np.array(features)
    new_data = pd.DataFrame([input_features], columns = cols)
    prediction = model.predict(new_data)
    return render_template("index.html", prediction_text='Player Class --> {}'.format(int(prediction[0])))

if __name__ == '__main__':
    app.run(debug=True)