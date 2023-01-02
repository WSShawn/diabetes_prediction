from urllib import request
import requests
import json
import pandas as pd
import random


# для отправки на сервер в json формате
def send_json(x):
    HighBP , HighChol ,CholCheck ,BMI ,Smoker ,\
    Stroke ,HeartDiseaseorAttack ,PhysActivity,Fruits ,\
    Veggies ,HvyAlcoholConsump ,\
    AnyHealthcare ,NoDocbcCost,GenHlth ,MentHlth ,\
    PhysHlth, DiffWalk,Sex , Age , Education , Income = x

    body = {
        'HighBP': HighBP , 'HighChol': HighChol ,
            'CholCheck':CholCheck ,'BMI':BMI ,'Smoker': Smoker ,
            'Stroke': Stroke ,'HeartDiseaseorAttack':HeartDiseaseorAttack ,
            'PhysActivity': PhysActivity,'Fruits': Fruits ,
            'Veggies': Veggies ,'HvyAlcoholConsump':HvyAlcoholConsump ,
            'AnyHealthcare': AnyHealthcare ,'NoDocbcCost':NoDocbcCost,
            'GenHlth': GenHlth ,'MentHlth': MentHlth ,
            'PhysHlth': PhysHlth, 'DiffWalk':DiffWalk,
            'Sex':Sex , 'Age': Age ,
            'Education': Education , 'Income':Income
    }

    myurl = 'http://127.0.0.1:5000/predict'
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    response = requests.post(myurl, json=body, headers=headers)
    return response.json()['Diagnosis']


# генерируем случайные подходящие данные
data_2 = list(random.randint(0, 1) for i in range(21))
data_2.pop(3)
data_2.insert(3, random.randint(20, 90))
final = tuple(data_2)

# получаем результат
res_2 = send_json(final)
print(res_2)