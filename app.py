from flask import Flask, render_template, request, jsonify
import json
import catboost
from catboost import CatBoostClassifier
import pandas as pd
import dill
import numpy as np


# Если мы собираем данные у пользователя

with open('Cat_655.dill', 'rb') as f:
    model = dill.load(f)

app = Flask(__name__, template_folder='template')


@app.route('/', methods=['GET'])
def say_hello():
    return render_template('predict_2.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    threshold = 0.215159
    HighBP, HighChol, CholCheck, BMI, Smoker, \
    Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, \
    Veggies, HvyAlcoholConsump, \
    AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, \
    PhysHlth, DiffWalk, Sex, Age, Education, Income = request.form.values()

    features = pd.DataFrame({
        'HighBP': [int(HighBP)], 'HighChol': [int(HighChol)],
        'CholCheck': [int(CholCheck)], 'BMI': [int(BMI)], 'Smoker': [int(Smoker)],
        'Stroke': [int(Stroke)], 'HeartDiseaseorAttack': [int(HeartDiseaseorAttack)],
        'PhysActivity': [int(PhysActivity)], 'Fruits': [int(Fruits)],
        'Veggies': [int(Veggies)], 'HvyAlcoholConsump': [int(HvyAlcoholConsump)],
        'AnyHealthcare': [int(AnyHealthcare)], 'NoDocbcCost': [int(NoDocbcCost)],
        'GenHlth': [int(GenHlth)], 'MentHlth': [int(MentHlth)],
        'PhysHlth': [int(PhysHlth)], 'DiffWalk': [int(DiffWalk)],
        'Sex': [int(Sex)], 'Age': [int(Age)],
        'Education': [int(Education)], 'Income': [int(Income)]
    })

    prediction = model.predict_proba(features)[:, 1][0]

    if prediction > threshold:
        return render_template('result.html', prediction_text='You have high risk to have diabetes')
    else:
        return render_template('result.html', prediction_text='You have low risk to have diabetes')


if __name__ == '__main__':
    app.run(debug=True)


