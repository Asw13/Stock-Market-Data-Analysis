import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from pickle import load
from pickle import dump
import warnings
warnings.filterwarnings('ignore')
import nsepy as nse
from datetime import date
import datetime
current_time=datetime.datetime.now()


# Background
# Set page config
st.set_page_config(page_title="My Streamlit App", page_icon=":smiley:", layout="wide")

# Set background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('white.webp')
    
# Add title with CSS styles
st.markdown("""
<style>
h1 {
    text-align: center;
    color:black;
}
</style>
""", unsafe_allow_html=True)

st.title("GROUP 5 Project")


# load the pre-trained LSTM model
loaded_model=pickle.load(open(r'bp_trained_model.sav','rb'))

bp1=nse.get_history(symbol='BERGEPAINT',start=date(2010,1,1),end=date(current_time.year,current_time.month,current_time.day))


bp=bp1[['Open','High','Low','Close']]

#describe data

st.markdown("<h2 style='color: red;'>Data from 2010 till today(describe data)</h2>", unsafe_allow_html=True)
st.write(bp.describe())

#visualization
st.markdown("<h2 style='color: red;'>Closing Price VS Time Chart</h2>", unsafe_allow_html=True)
fig = plt.figure(figsize = (12,6))
plt.plot(bp.Close)
st.pyplot(fig)

st.markdown("<h2 style='color: red;'>Closing Price VS Time Chart with 100MA</h2>", unsafe_allow_html=True)
ma100 =bp.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(bp.Close)
st.pyplot(fig)

st.markdown("<h2 style='color: red;'>Closing Price VS Time Chart with 100MA & MA200</h2>", unsafe_allow_html=True)
ma100 =bp.Close.rolling(100).mean()
ma200 =bp.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(bp.Close,'b')
st.pyplot(fig)




scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(bp)

def bp_stockprice(n_future):
    global pred_future
    #n_future = input()  # Number of days to forecast
    n_past = 60
    last_30_days = scaled_data[-n_past:]
    X_test = np.array([last_30_days])
    predictions = []

    for i in range(n_future):
        next_day_pred = loaded_model.predict(X_test)[0, 0]
        last_30_days = np.append(last_30_days[1:, :], [[next_day_pred, next_day_pred, next_day_pred, next_day_pred]], axis=0)
        X_test = np.array([last_30_days])
        pred_value = scaler.inverse_transform([[0, 0, 0, next_day_pred]])[0, 3]
        predictions.append(pred_value)
        print("Day {}: {}".format(i+1, pred_value))
        
    
    start_date = datetime.datetime(current_time.year,current_time.month,current_time.day)
    dates = [start_date + datetime.timedelta(days=idx) for idx in range(n_future)]
    
    data1 = {
        
        
        'predictions': predictions
        }
  
    pred_future= pd.DataFrame(data1,dates)
    #pred_future.set_index('Date', inplace=True)
    
    return np.round(pred_future,0)
    
def plo():
 st.markdown("<h2 style='color: red;'>predictions plot</h2>", unsafe_allow_html=True)
 fig2 = plt.figure(figsize=(12,6))
 plt.plot(bp[['Close']], 'b', label = "Original Price")
 plt.plot(pred_future, label = "Predicted Price")
 plt.xlabel('Time')
 plt.ylabel('Price')
 plt.legend()
 plt.show()
 st.pyplot(fig2)


def main():
    #giving title
    st.title('Forcasting future data web app')

    #getting input variable from users
    n_future=st.text_input('Number of future data')

    diagnosis=''

    #creating button for prediction
    if st.button('Future days data predicted'):
        diagnosis=bp_stockprice(int(n_future))
    st.write(diagnosis)

    plo()
  
if __name__ == '__main__':
    main()


