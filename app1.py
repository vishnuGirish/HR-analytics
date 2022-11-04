#import relevant libraries for flask, html rendering and loading the ML model
from flask import Flask,request, url_for, redirect, render_template
import pickle
import joblib
import pandas as pd

app = Flask(__name__)

model=joblib.load('model.pkl')
scaled=joblib.load('scaled.pkl')
le_dept=joblib.load('le_dept.pkl')
le_edu=joblib.load('le_edu.pkl')
le_gen=joblib.load('le_gen.pkl')
le_rech=joblib.load('le_rech.pkl')
le_reg=joblib.load('le_reg.pkl')

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
def predict():
    
    Department = request.form['1']
    Department = le_dept.transform([Department])[0]
    Education = request.form['2']
    Education = le_edu.transform([Education])[0]
    Gender = request.form['3']
    Gender = le_gen.transform([Gender])[0]
    Recruitment_channel = request.form['4']
    Recruitment_channel = le_rech.transform([Recruitment_channel])[0]
    No_of_trainings = request.form['5']
    Age = request.form['6']
    Previous_year_rating = request.form['7']
    Length_of_service = request.form['8']
    KPIs_met_above_80percent= request.form['9']
    Awards_won = request.form['10']
    Avg_training_score = request.form['11']
    Region = request.form['12']
    Region = le_reg.transform([Region])[0]
    

    rowDF= pd.DataFrame([pd.Series([Department,Education,Gender,Recruitment_channel,No_of_trainings,Age,Previous_year_rating,Length_of_service,
    KPIs_met_above_80percent,Awards_won,Avg_training_score,Region])])
    rowDF_new=pd.DataFrame(scaled.transform(rowDF))

    print(rowDF_new)

    #  model prediction 
    prediction= model.predict_proba(rowDF_new)
    print(f"The  Predicted values is :{prediction[0][1]}")

    if prediction[0][1] >=0.5:
        valPred = round(prediction[0][1],3)
        print(f"The Round val {valPred*100}%")
        return render_template('result.html',pred=f'Probability of being Promoted  is {valPred*100}%.')
    else:
        valPred = round(prediction[0][0],3)
        return render_template('result.html',pred=f' Probability of not being Promoted is {valPred*100:.2f}%.')

if __name__ == '__main__':
    app.run(debug=True)