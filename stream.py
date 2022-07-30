from fileinput import filename
import joblib
import numpy as np
import streamlit as st

filename = './titan.pkl'
model = joblib.load(filename)
# model = pickle.load(open('C:/Users/pc/Desktop/mydataanalysis/python/jupyterNotebook/project/Titanic-App/titan.pkl','rb'))

def main():
  st.title('Titanic prediction AI APP')
  st.image('./titanic.jpg', 'pixelstalk.net')
  st.header('Surviving the Titanic?')
  st.write("Kindly fill the inputs accordingly with the passenger's boarding details.")
  # input variables
  Passenger_id = st.text_input('Passenger_id (1-1309)')
  Pclass = st.text_input('passenger class (1-3)')
  sex = st.selectbox('Sex', ['Select','male', 'female'])
  Age = st.text_input('Age (1-80)')
  sibs = st.text_area('No. sibling (0-8)')
  parch = st.text_input('No. of parent or children (1-9)')
  Fare = st.text_input('Fare (0-512)')
  Embarked = st.selectbox('Embarked port',['Select','Q','C','S'])

  # prediction code
  if st.button('Predict'):
    makeprediction = model.predict([[Passenger_id,Pclass,sex,Age,sibs,parch,Fare,Embarked]])
    output = round(makeprediction[0], 2)
    if output == 1:
      st.success('This passenger survived😊')
    else:
      st.success('This passenger died😔')
  st.title('The Titanic App')
  st.write("""
      The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, 
      during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after 
      colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard,
      resulting in the death of 1502 out of 2224 passengers and crew. While there was some element of 
      luck involved in surviving,
      it seems some groups of people were more likely to survive than others.
      This Full-Stack AI app was built based on a predictive Machine Learning Model that answers 
      the question: “what sorts of people were more likely to survive?” using passenger data 
      (ie name, age, gender, socio-economic class, etc).
      The App is an AI application trained using Random Forest Classifie. The data-set trained was 
      gotten from Kaggle datasets(link below). The exploratory data analysis can be found in this  notebook(link below) and
       the codes can be found in this repository(link below)

    Note for user:Select or Input correct information where necessary
  """) 
  st.write("[notebook](https://github.com/kennytheanalystt)", unsafe_allow_html=True)
  st.write("[Kaggle data-set](https://www.kaggle.com/competitions/titanic/data)", unsafe_allow_html=True)
  st.write("[Code](https://github.com/kennytheanalystt)", unsafe_allow_html=True)
if __name__ == '__main__':
  main()