# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Prediksi Saham (04201003)')

stocks = ('INTC', 'NVDA', 'AMD', 'GME','IBM','ORCL','IFX','MU','MSFT')
selected_stock = st.selectbox('Pilih saham untuk diprediksi', stocks)

n_years = st.slider('Tahun prediksi:', 1, 5)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Memuat data...')
data = load_data(selected_stock)
data_load_state.text('Memuat data... selesai!')

st.subheader('Data mentah')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="saham_buka"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="saham_tutup"))
	fig.layout.update(title_text='Silder data seri waktu', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Data forecast')
st.write(forecast.tail())
    
st.write(f'Forecast plot untuk {n_years} tahun')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Komponen forecast")
fig2 = m.plot_components(forecast)
st.write(fig2)
