# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
df = pd.read_csv('Titanic_train.csv')

# Preprocessing
# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Features & target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])

    data = [[pclass, sex, age, sibsp, parch, fare]]
    prediction = model.predict(data)[0]

    result = "Survived" if prediction == 1 else "Did Not Survive"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)


