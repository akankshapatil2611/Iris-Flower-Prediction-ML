import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# create flask app
app = Flask(__name__)

# Load the Pickle Model
model = pickle.load(open("i_model.pkl", 'rb'))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    if prediction==[0]:
        p = 'Iris-setosa'
    elif prediction == [1]:
        p = 'Iris-versicolor'
    else:
        p = 'Iris-virginica'

    return render_template("index.html", prediction_text="The Flower Species is {}".format(p))

if __name__ == "__main__":
    app.run(debug=True)