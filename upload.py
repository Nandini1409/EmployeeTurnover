# page1.py

import streamlit as st 
st.markdown("""
<style>
    .stButton > button {
    border-radius:20px;
    font-size:22px;

}
    .stButton:hover > button{
        color:rgb(255, 75, 75);
        border-color:rgb(255, 75, 75);
    }
 
    h4:hover{
        color:rgb(255, 75, 75);
};
    }
</style>
    """, unsafe_allow_html=True)

import pandas as pd

original_title = '<marquee style="color:lime"> <h4> !!! Welcome To Employee Turnover Prediction System !!! </h4></marquee>'
st.markdown(original_title, unsafe_allow_html=True)

question = ' What is Employee Turnover ?'
st.subheader(question)

para1='<span><p>Employee Turnover or Employee Turnover ratio is the measurement of the total number of employees who leave an organization in a particular year. Employee Turnover Prediction means to predict whether an employee is going to leave the organization in the coming period.</p></span>'
st.markdown(para1, unsafe_allow_html=True)
para2='<span><p>A Company uses this predictive analysis to measure how many employees they will need if the potential employees will leave their organization. A company also uses this predictive analysis to make the workplace better for employees by understanding the core reasons for the high turnover ratio.</p></span>'
st.markdown(para2, unsafe_allow_html=True)
st.subheader('Data Preprocessing')
para3='<span><p><b>Now let’s dive into the data to move further with this project on Employee Turnover Prediction.</b></p></span>'
st.markdown(para3, unsafe_allow_html=True)

from io import StringIO 
# Load csv data into numpy array, skip header row and specify data types.
file = st.file_uploader(":Please choose a Dataset To Proceed Further:")

if file is not None:

    #To read file as bytes:
    bytes_data = file.getvalue()
    
    #To convert to a string based IO:
    stringio = StringIO(file.getvalue().decode("utf_8"))
    
    #To read file as string:
    string_data = stringio.read()
    data= pd.read_csv(file)

    st.subheader("The Number of data Present In dataset is figured below ")
    st.subheader(data.shape)
    st.table(data [:5])


import streamlit as st
import webbrowser

import time
with st.spinner('wait content loading'):
    time.sleep(8)

url = 'http://192.168.43.126:8501'
st.subheader("For Visualization and Prediction Click onto Next Button")
if st.button('Next'):
    webbrowser.open_new_tab(url)
