import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle
import base64
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#from tensorflow.python.keras.layers.recurrent import LSTM,Dense
import keras
from keras import layers
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from pickle import load
from pickle import dump
import warnings
warnings.filterwarnings('ignore')
from nselib import capital_market
import nsepy as nse
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from datetime import date
import datetime

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

# st.title("GROUP 5 Project")
st.markdown(
    """
    <style>
    .blink {
        animation: blinker 2s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    .grey-box {
        border: 2px solid grey;
        border-radius: 5px;
        padding: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """
, unsafe_allow_html=True)

st.markdown('<div class="grey-box blink"><h1>GROUP 5 Project</h1></div>', unsafe_allow_html=True)

# load the pre-trained LSTM model
loaded_model=pickle.load(open(r'bp_trained_model.sav','rb'))
current_time=datetime.datetime.now()
formatted_time = current_time.strftime("%d-%m-%Y")
formatted_time

bp1= capital_market.price_volume_and_deliverable_position_data(symbol='BERGEPAINT', from_date='01-01-2010', to_date=formatted_time)

#bp1=nse.get_history(symbol='BERGEPAINT',start=date(2010,1,1),end=date(current_time.year,current_time.month,current_time.day))
bp2=bp1[['Date','ClosePrice']]
column_mapping = {'Date':'Date','ClosePrice':'Close'}
bp2.rename(columns=column_mapping,inplace=True)
bp2['Date']= pd.to_datetime(bp2['Date'])
#we are considering columns 
bp=bp1[['OpenPrice','HighPrice','LowPrice','ClosePrice']]
column_mapping = {'OpenPrice': 'Open', 'HighPrice': 'High','LowPrice':'Low','ClosePrice':'Close'}
# cahnge bp column name to Column mapping

bp.rename(columns=column_mapping,inplace=True)

#bp=bp1[['Open','High','Low','Close']]

#describe data

st.markdown("<h2 style='color: red;'>Data from 2010 till today(describe data)</h2>", unsafe_allow_html=True)
st.write(bp2.head())
st.write(bp2.describe())

#visualization

st.markdown("<h2 style='color: red;'>Closing Price VS Time Chart</h2>", unsafe_allow_html=True)
fig=plt.figure(figsize=(14,5))
sns.set_style("ticks")
sns.lineplot(data=bp2,x="Date",y='Close',color='firebrick')
sns.despine()
plt.title("The Stock Price of BERGEPAINT ",size='x-large',color='blue')
st.pyplot(fig)

ma100 =bp.Close.rolling(100).mean()
st.markdown("<h2 style='color: red;'>Closing Price VS Time Chart with 100MA</h2>", unsafe_allow_html=True)
fig=plt.figure(figsize=(14,5))
sns.set_style("ticks")
sns.lineplot(data=bp2,x="Date",y=bp.Close.rolling(100).mean(),color='blue')
sns.lineplot(data=bp2,x="Date",y='Close',color='firebrick')
sns.despine()
plt.title("The Stock Price of BERGEPAINT ",size='x-large',color='blue')
st.pyplot(fig)


st.markdown("<h2 style='color: red;'>Closing Price VS Time Chart with 100MA & MA200</h2>", unsafe_allow_html=True)
fig=plt.figure(figsize=(14,5))
sns.set_style("ticks")
sns.lineplot(data=bp2,x="Date",y=bp.Close.rolling(100).mean(),color='green')
sns.lineplot(data=bp2,x="Date",y=bp.Close.rolling(200).mean(),color='blue')
sns.lineplot(data=bp2,x="Date",y='Close',color='firebrick')
sns.despine()
plt.title("The Stock Price of BERGEPAINT ",size='x-large',color='blue')
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
 fig3=plt.figure(figsize=(14,5))
 sns.set_style("ticks")
 sns.lineplot(data=bp2,x="Date",y='Close',color='firebrick',label="Original Price")
 sns.lineplot(data=pred_future,color='Green',label='Predicted Price')
 sns.despine()
 plt.title("The Stock Price of BERGEPAINT ",size='x-large',color='blue')
 st.pyplot(fig3)
 


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


