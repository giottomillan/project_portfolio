import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplcyberpunk
import numpy as np
from prophet import Prophet
import yfinance as yf
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go




plt.style.use("dark_background")

###########################
#### Funciones Principales
###########################



def get_data(stock, start_time, end_time):
    data = yf.download(stock, start=start_time, end=end_time)
    data = data.sort_index(ascending=False)
    return data


def get_levels(dfvar):
        

    def isSupport(df,i):
        support = df['low'][i] < df['low'][i-1]  and df['low'][i] < df['low'][i+1] and df['low'][i+1] < df['low'][i+2] and df['low'][i-1] < df['low'][i-2]
        return support

    def isResistance(df,i):
        resistance = df['high'][i] > df['high'][i-1]  and df['high'][i] > df['high'][i+1] and df['high'][i+1] > df['high'][i+2] and df['high'][i-1] > df['high'][i-2]
        return resistance

    def isFarFromLevel(l, levels, s):
        level = np.sum([abs(l-x[0]) < s  for x in levels])
        return  level == 0
    
    
    df = dfvar.copy()
    df.rename(columns={'High':'high','Low':'low'}, inplace=True)
    s =  np.mean(df['high'] - df['low'])
    levels = []
    for i in range(2,df.shape[0]-2):
        if isSupport(df,i):  
            levels.append((i,df['low'][i]))
        elif isResistance(df,i):
            levels.append((i,df['high'][i]))

    filter_levels = []
    for i in range(2,df.shape[0]-2):
        if isSupport(df,i):
            l = df['low'][i]
            if isFarFromLevel(l, levels, s):
                filter_levels.append((i,l))
        elif isResistance(df,i):
            l = df['high'][i]
            if isFarFromLevel(l, levels, s):
                filter_levels.append((i,l))

    return filter_levels

def plot_close_price(data):
    # Assuming get_levels is defined elsewhere and returns levels data
    levels = get_levels(data)
    df_levels = pd.DataFrame(levels, columns=['index','close'])
    df_levels.set_index('index', inplace=True)
    max_level = df_levels.idxmax()
    min_level = df_levels.idxmin()

    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

    if min_level.close > max_level.close:
        trend = 'down'
        fib_levels = [data.Close.iloc[max_level.close] - (data.Close.iloc[max_level.close] - data.Close.iloc[min_level.close]) * ratio for ratio in ratios]
        idx_level = max_level
    else:
        trend = 'up'
        fib_levels = [data.Close.iloc[min_level.close] + (data.Close.iloc[max_level.close] - data.Close.iloc[min_level.close]) * ratio for ratio in ratios]
        idx_level = min_level

    # Creating the plot
    fig = go.Figure()

    # Add the Close price line
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'],
        mode='lines',
        line=dict(color='dodgerblue', width=2),
        name='Close Price'
    ))

    # Add Fibonacci levels
    for level, ratio in zip(fib_levels, ratios):
        fig.add_trace(go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[level, level],
            mode='lines',
            line=dict(color='red', dash='dot', width=0.9),
            name="{:.1f}%".format(ratio * 100)
        ))

    # Update layout to match the desired style
    fig.update_layout(
        title='Close Price with Fibonacci Levels',
        xaxis_title='Date',
        yaxis_title='Precio USD',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='red'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.2),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.2),
        images=[dict(
            source='./images/Leonardo_Diffusion_Generate_a_captivating_and_professional_log_1.jpg',
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            sizex=0.4, sizey=0.4,
            xanchor="center", yanchor="middle",
            opacity=0.15,
            layer="below"
        )]
    )
    return fig


def daily_returns(df):
    df = df.sort_index(ascending=True)
    df['returns'] = np.log(df['Close']).diff()
    return df

def returns_vol(df):
    df['volatility'] = df.returns.rolling(12).std()
    return df

def plot_volatility(df_vol):
    # Create the plot
    fig = go.Figure()

    # Add the returns line
    fig.add_trace(go.Scatter(
        x=df_vol.index,
        y=df_vol['returns'],
        mode='lines',
        line=dict(color='dodgerblue', width=0.5),
        name='Retornos Diarios'
    ))

    # Add the volatility line
    fig.add_trace(go.Scatter(
        x=df_vol.index,
        y=df_vol['volatility'],
        mode='lines',
        line=dict(color='darkorange', width=1),
        name='Volatilidad Móvil'
    ))

    # Update layout to match the desired style
    fig.update_layout(
        title='Volatilidad y Retornos Diarios',
        xaxis_title='Fecha',
        yaxis_title='% Porcentaje',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.2),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.2),
        yaxis_tickformat=".3f",
        images=[dict(
            source='./images/Leonardo_Diffusion_Generate_a_captivating_and_professional_log_1.jpg',
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            sizex=0.4, sizey=0.4,
            xanchor="center", yanchor="middle",
            opacity=0.15,
            layer="below"
        )]
    )
    return fig


def plot_prophet(data, n_forecast=365):
    # Prepare the data for Prophet
    data_prophet = data.reset_index().copy()
    data_prophet.rename(columns={'Date':'ds','Close':'y'}, inplace=True)

    # Fit the Prophet model
    m = Prophet()
    m.fit(data_prophet[['ds', 'y']])

    # Make future predictions
    future = m.make_future_dataframe(periods=n_forecast)
    forecast = m.predict(future)

    # Create the plot
    fig = go.Figure()

    # Add the historical data
    fig.add_trace(go.Scatter(
        x=data_prophet['ds'],
        y=data_prophet['y'],
        mode='lines',
        line=dict(color='dodgerblue', width=1),
        name='Histórico'
    ))

    # Add the forecast data
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        line=dict(color='darkorange', width=0.5),
        name='Predicción'
    ))

    # Add uncertainty intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(color='gray', dash='dash', width=0.5),
        name='Intervalo Superior'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(color='gray', dash='dash', width=0.5),
        fill='tonexty',
        name='Intervalo Inferior'
    ))

    # Update layout to match the desired style
    fig.update_layout(
        title='Predicción de Precio de Cierre',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.2),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.2),
        images=[dict(
            source='./images/Leonardo_Diffusion_Generate_a_captivating_and_professional_log_1.jpg',
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            sizex=0.4, sizey=0.4,
            xanchor="center", yanchor="middle",
            opacity=0.15,
            layer="below"
        )]
    )

    return fig

# Crear el gráfico inicial (de reserva)
def plot_open_close(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x=data.index, y='Open', color='yellow', linewidth=1, label='Open')
    sns.lineplot(data=data, x=data.index, y='Close', color='red', linewidth=1, label='Close')
    sns.lineplot(data=data, x=data.index, y='High', color='green', linewidth=1, label='High')
    sns.lineplot(data=data, x=data.index, y='Low', color='dodgerblue', linewidth=1, label='Low')
    mplcyberpunk.add_glow_effects()
    plt.title(f'Precio de Apertura y Cierre para {stock}')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.grid(True,color='gray', linestyle='-', linewidth=0.4)
    return plt


###########################
#### LAYOUT - Sidebar
###########################

logo_pypro = Image.open('./images/Leonardo_Diffusion_Generate_a_captivating_and_professional_log_1.jpg')
# Lists of stocks and cryptocurrencies
stocks = ['NKE','^RUT', '^GSPC', '^IXIC','^DJI','NVDA','TSLA','MSFT','AMZN','INTC','AMD','JNJ','BABA','GOOGL','QCOM', 'AAPL', 'META', 'NFLX', 'SPOT']
cryptocurrencies = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'DOGE-USD', 'ADA-USD', 'XLM-USD', 'MANA-USD', 'ALGO-USD', 'ATOM-USD', 'DOT-USD']

with st.sidebar:
    st.image(logo_pypro)
    
    # Select between Stock or Cryptocurrency
    asset_type = st.radio("Select Asset Type", ('Stocks', 'Cryptocurrencies'))
    
    if asset_type == 'Stocks':
        stock = st.selectbox('Select Stock Ticker', stocks, index=1)
    else:
        stock = st.selectbox('Select Cryptocurrency Ticker', cryptocurrencies, index=0)
    
    start_time = st.date_input("Start Date", datetime.date(2019, 7, 6))
    end_time = st.date_input("End Date", datetime.date.today())
    periods = st.number_input('Forecast Periods', value=365, min_value=1, max_value=5000)


###########################
#### DATA - Funciones sobre inputs
###########################

data = get_data(stock, start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"))
plot_price = plot_close_price(data)

df_ret = daily_returns(data)
df_vol = returns_vol(df_ret)
plot_vol = plot_volatility(df_vol)

plot_forecast = plot_prophet(data, periods)

# Función para obtener los datos financieros
def financial_data(stock):
    data = yf.Ticker(stock)
    df = data.get_income_stmt()  # Cambia get_income_stmt() a financials para obtener los datos financieros
    return df

# Función para obtener los datos contables
def contable_data(stock):
    data = yf.Ticker(stock)
    df = data.balance_sheet  # Cambia get_income_stmt() a financials para obtener los datos financieros
    return df

#Función para de cálculo del PER
# Function to calculate the PER ratio with error handling
def per_ratio(stock):
    try:
        info = yf.Ticker(f'{stock}')
        hist = info.history()
        
        # Ensure there is historical data to avoid indexing errors
        if hist.empty:
            raise ValueError("No historical data available for the stock")

        market_value_per_share = hist.Close.iloc[0]

        income_stmt = info.income_stmt.T

        # Ensure 'Basic EPS' is available in the income statement
        if 'Basic EPS' not in income_stmt.columns:
            raise ValueError("'Basic EPS' not found in the income statement")

        EPS = income_stmt['Basic EPS'].iloc[0]

        # Ensure EPS is not zero to avoid division by zero error
        if EPS == 0:
            raise ValueError("EPS is zero, cannot calculate PER")

        per = market_value_per_share / EPS
        return per
    except Exception as e:
        return f"Error calculating PER: {e}"
     
#precio de la acción
# Function to get the market value per share
def precio(stock):
    try:
        # Fetch historical data using yf.download
        hist = yf.download(stock)
        
        if hist.empty:
            raise ValueError("No historical data available for the stock")
        
        # Sort the historical data by date in descending order
        hist = hist.sort_values('Date', ascending=False)
        
        market_value_per_share = hist.Close.iloc[0]
        
        return market_value_per_share

    except ValueError as ve:
        return f"ValueError: {ve}"
    except IndexError as ie:
        return "IndexError: Could not retrieve the market value per share, historical data might be incomplete"
    except Exception as e:
        return f"An error occurred: {e}"

def plotly_image(data):
    # Crear el gráfico de dispersión
    fig = px.line(data, x=data.index, y=data.columns, color_discrete_sequence=px.colors.qualitative.Plotly)

    # Personalizar el gráfico para un estilo cyberpunk
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.2),
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=0.2),
        title={
            'text': "Precio de la Acción",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return fig

###########################
#### LAYOUT - Render Final
###########################

st.title("Análisis de Acciones")

st.subheader(f'El precio actual de {stock}')
st.write(precio(stock))

st.subheader(f'El PER de {stock}')
st.write(per_ratio(stock))

st.subheader('Precio de Apertura y Cierre')
st.plotly_chart(plotly_image(data))

st.subheader('Precio de Cierre - Fibonacci')
st.plotly_chart(plot_price)

st.subheader(f'Datos Históricos de {stock}')
st.dataframe(data)

st.subheader('Retornos Diarios')
st.plotly_chart(plot_vol)

st.subheader(f'Datos Financieros de {stock}')
st.dataframe(financial_data(stock))

st.subheader(f'Datos Contables de {stock}')
st.dataframe(contable_data(stock))

st.subheader('Forecast a un Año - Prophet')
st.plotly_chart(plot_forecast)









