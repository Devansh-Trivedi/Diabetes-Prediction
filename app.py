import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r'https://raw.githubusercontent.com/Devansh-Trivedi/Diabetes-Prediction/main/diabetes.csv')
X = dataset.iloc[:, [1, 3, 4, 5, 7]].values
Y = dataset.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0, stratify = dataset['Outcome'] )
random_forest_model = RandomForestClassifier(n_estimators = 11, random_state = 0)
random_forest_model.fit(X_train, Y_train)

def user_report():
  Glucose = st.sidebar.slider('Glucose', 0,200, 136 )
  SkinThickness = st.sidebar.slider('Skin Thickness', 0,100, 0 )
  Insulin = st.sidebar.slider('Insulin', 0,846, 0 )
  BMI = st.sidebar.slider('BMI', 0,67, 31 )
  Age = st.sidebar.slider('Age', 21,88, 22 )

  user_report_data = {
      'Glucose':Glucose,
      'SkinThickness':SkinThickness,
      'Insulin':Insulin,
      'BMI':BMI,
      'Age':Age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
user_result = random_forest_model.predict( user_data )
print(user_result)
st.title('Your Report: ')
output=''
if int(user_result[0])==0:
  output = 'You are not Diabetic'
  color = 'blue'
else:
  output = 'You are Diabetic'
  color = 'red'
st.subheader(output)

def glucose_vs_age_graph():
  st.header('Glucose Value Graph (Others vs Yours)')
  glucose_vs_age = plt.figure()
  sns.scatterplot(x = 'Age', y = 'Glucose', data = dataset, hue = 'Outcome' , palette='magma')
  sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,220,10))
  plt.title('0 - Not Diabetic, 1 - Diabetic')
  st.pyplot(glucose_vs_age)

def skin_vs_age_graph():
  st.header('Skin Thickness Value Graph (Others vs Yours)')
  skin_vs_age = plt.figure()
  sns.scatterplot(x = 'Age', y = 'SkinThickness', data = dataset, hue = 'Outcome', palette='Blues')
  sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,110,10))
  plt.title('0 - Not Diabetic, 1 - Diabetic')
  st.pyplot(skin_vs_age)

def insulin_vs_age_graph():
  st.header('Insulin Value Graph (Others vs Yours)')
  insulin_vs_age = plt.figure()
  sns.scatterplot(x = 'Age', y = 'Insulin', data = dataset, hue = 'Outcome', palette='rocket')
  sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,900,50))
  plt.title('0 - Not Diabetic, 1 - Diabetic')
  st.pyplot(insulin_vs_age)

def bmi_vs_age_graph():
  st.header('BMI Value Graph (Others vs Yours)')
  bmi_vs_age = plt.figure()
  sns.scatterplot(x = 'Age', y = 'BMI', data = dataset, hue = 'Outcome', palette='rainbow')
  sns.scatterplot(x = user_data['Age'], y = user_data['BMI'], s = 150, color = color)
  plt.xticks(np.arange(10,100,5))
  plt.yticks(np.arange(0,70,5))
  plt.title('0 - Not Diabetic, 1 - Diabetic')
  st.pyplot(bmi_vs_age)


st.title('Your Report Graphs:')
glucose_vs_age_graph()
skin_vs_age_graph()
insulin_vs_age_graph()
bmi_vs_age_graph()

st.title('Accuracy: ')
st.write(str(accuracy_score(Y_test, random_forest_model.predict(X_test))*100)+'%')
