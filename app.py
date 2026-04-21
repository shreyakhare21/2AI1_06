# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
df = pd.read_csv('Titanic_train.csv')

# Preprocessing
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [[
        int(request.form['pclass']),
        int(request.form['sex']),
        float(request.form['age']),
        int(request.form['sibsp']),
        int(request.form['parch']),
        float(request.form['fare'])
    ]]

    prediction = model.predict(data)[0]
    result = "Survived 🎉" if prediction == 1 else "Did Not Survive ❌"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)


