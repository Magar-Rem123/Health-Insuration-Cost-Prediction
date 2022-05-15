# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template
model = joblib.load(open('insurance_cost.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods = ['POST'])
def predict():
    data1 = request.form['age']
    data2 = request.form['gender']
    data3 = request.form['bmi']
    data4 = request.form['smoker']
    data5 = request.form['children']
    data6 = request.form['region']
    arr = np.array([[data1,data2,data3,data4,data5,data6]])
    pred = model.predict(arr)
    output = np.round(pred[0],2)
    return render_template('index.html',prediction_text = "The insuration  cost is:{}".format(output))
if __name__=='__main__':
    app.run()

