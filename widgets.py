import streamlit as st
import pandas as pd

st.title("Streamlit text input")
name = st.text_input("Enter your name", "Type here...")

if name:
    st.write(f"Hello, {name}!")

# This is for the slider to select your age
age = st.slider("Select your age", 0, 100, 25)
st.write(f"You are {age} years old.")

option = ['Java', 'C++', 'Python', 'Ruby']
language = st.selectbox("Select your programming language", option)
st.write(f"Your favourite programming language is {language}")

#Now lets make a dataframe and save the data as csv

df = pd.DataFrame(
    {
        "Name": ["Rahul", "Kanupriya", "Raavika", "Shashi", "Poonam"],
        "Age": [34, 34, 4, 57, 61]
    }
)
df.to_csv(r"D:\ChatbotPreparation\abc.csv")
#Now load the csv and show the contents of that on the page

file_uploader = st.file_uploader("Choose the file to upload", type="csv")
if file_uploader:
    df = pd.read_csv(file_uploader)
    st.write("The contents of the file are ")
    st.write(df)