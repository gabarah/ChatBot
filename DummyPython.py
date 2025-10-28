import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
st.title("This is my first chatbot title")
st.write("This is a dummy Python script for testing purposes.")
st.write("Current working directory:", os.getcwd())
print(sys.executable)
# Here i made a dataframe and published that to the streamlit app
df = pd.DataFrame({
    'FirstColumn': [1, 2, 3, 4],
    'SecondColumn': [10, 20, 30, 40]
})

st.write(df)

# I am making a 20*3 dataframe and making a line chart of it with column names given to the same
chart_data = pd.DataFrame(
    np.random.randn(20,3), columns=['a', 'b', 'c']  
)
st.line_chart(chart_data)
