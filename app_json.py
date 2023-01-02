from flask import Flask, render_template, request, jsonify
import json
import catboost
from catboost import CatBoostClassifier
import pandas as pd
import dill
import numpy as np
# если данные уже есть в json формате
with open('Cat_655.dill', 'rb') as f:
    model = dill.load(f)


app = Flask(__name__, template_folder='template')


@app.route('/', methods=['GET'])
def say_hello():
    return 'Diabetes prediction'


@app.route('/predict', methods=['POST'])
def predict():
    threshold = 0.215159
    data = request.get_json()

    HighBP = data['HighBP']
    HighChol = data['HighChol']
    CholCheck = data['CholCheck']
    BMI = data['BMI']
    Smoker = data['Smoker']
    Stroke = data['Stroke']
    HeartDiseaseorAttack = data['HeartDiseaseorAttack']
    PhysActivity = data['PhysActivity']
    Fruits = data['Fruits']
    Veggies = data['Veggies']
    HvyAlcoholConsump = data['HvyAlcoholConsump']
    AnyHealthcare = data['AnyHealthcare']
    NoDocbcCost = data['NoDocbcCost']
    GenHlth = data['GenHlth']
    MentHlth = data['MentHlth']
    PhysHlth = data['PhysHlth']
    DiffWalk = data['DiffWalk']
    Sex = data['Sex']
    Age = data['Age']
    Education = data['Education']
    Income = data['Income']

    features = pd.DataFrame({
        'HighBP': [HighBP] , 'HighChol': [HighChol] ,
            'CholCheck':[CholCheck] ,'BMI':[BMI] ,'Smoker': [Smoker] ,
            'Stroke': [Stroke] ,'HeartDiseaseorAttack':[HeartDiseaseorAttack] ,
            'PhysActivity': [PhysActivity],'Fruits': [Fruits] ,
            'Veggies': [Veggies] ,'HvyAlcoholConsump':[HvyAlcoholConsump] ,
            'AnyHealthcare': [AnyHealthcare] ,'NoDocbcCost':[NoDocbcCost],
            'GenHlth': [GenHlth] ,'MentHlth': [MentHlth ],
            'PhysHlth': [PhysHlth], 'DiffWalk':[DiffWalk],
            'Sex':[Sex] , 'Age': [Age] ,
            'Education': [Education] , 'Income':[Income]
    })
    prediction = model.predict_proba(features)[:, 1]
    if prediction > threshold:
        result = 'You have high risk to have diabetes'
    else:
        result = 'You have low risk to have diabetes'

    return jsonify({'Diagnosis': result})


if __name__ == '__main__':
    app.run(debug=True)
