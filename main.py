import flask
from pickle import load
from numpy import array, shape
import sklearn.preprocessing

#region Global Variables

UI_FILE = 'index.html'
THRESHOLD = 0.75

#endregion

app = flask.Flask(__name__)

# Loading our trained model.
with open('neural_network_model.pkl', 'rb') as file:
    model = load(file)

# Create StandardScaler instance
scaler = sklearn.preprocessing.StandardScaler()

@app.route('/')
def home():
    return flask.render_template(UI_FILE)

@app.route('/results', methods = ['POST'])
def predict():

    # Fetching the features given by the user.
    features = [
        float(flask.request.form['First-Term-GPA']),
        float(flask.request.form['Second-Term-GPA']),
        int(flask.request.form['First-Language']),
        int(flask.request.form['Funding']),
        int(flask.request.form['Fast Track']),
        int(flask.request.form['Co-Op']),
        int(flask.request.form['Residency']),
        int(flask.request.form['Gender']),
        int(flask.request.form['Prev Education']),
        int(flask.request.form['Age Group']),
        float(flask.request.form['Math Score']),
        int(flask.request.form['English Grade'])
    ]
    
    # Reshaping the array as a part of data pre-processing for predicting the results.
    features_array = array(features).reshape(1, -1)

    # Scaling the features between the range of 0 and 1.
    # features_scaled = scaler.fit_transform(X = features_array)
    
    # Giving model a result to generate based on the given data by the user.
    prediction = model.predict(features_array)
    print(prediction)
 
    # Generating the results based on the configured threshold.
    result = (prediction > THRESHOLD).astype(int)
    
    # Sending the predicted results to the UI.
    return flask.render_template(UI_FILE, result = result[0][0])

if __name__ == '__main__':
    app.run(debug = True)