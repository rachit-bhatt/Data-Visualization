from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pickled model
with open('neural_network_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create StandardScaler instance
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    features = [
        float(request.form['First_Term_Gpa']),
        float(request.form['Second_Term_Gpa']),
        int(request.form['First_Language']),
        int(request.form['Funding']),
        int(request.form['Fast_Track']),
        int(request.form['Coop']),
        int(request.form['Residency']),
        int(request.form['Gender']),
        int(request.form['Prev_Education']),
        int(request.form['Age_Group']),
        float(request.form['Math_Score']),
        int(request.form['English_Grade'])
    ]

    # Convert features to numpy array and reshape
    features_array = np.array(features).reshape(1, -1)

    # Apply StandardScaler
    features_scaled = scaler.fit_transform(features_array)

    # Make prediction
    prediction = model.predict(features_scaled)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug = True)