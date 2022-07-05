# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 18:10:20 2022

@author: Admin
"""

import numpy as np
from collections.abc import Mapping
from flask import Flask, request, jsonify, render_template
import pickle

# flask app
app = Flask(__name__)
# loading model
model = pickle.load(open('model_final.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict' ,methods = ['POST'])
def predict():
    final_features = [float(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)
    
    return render_template('index.html', output='Agent will reach in {} minutes'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)