import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl","rb"))

# Load scaler
scaler = pickle.load(open("scaler.pkl","rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    result = model.predict(new_data)
    print(result[0])
    return jsonify(result[0])

@app.route("/predict",methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    new_data = scaler.transform(np.array(data).reshape(1,-1))
    result = model.predict(new_data)
    print(result[0])
    return render_template("index.html",prediction_text=result[0])

if __name__ == "__main__":
    app.run(debug=True)
