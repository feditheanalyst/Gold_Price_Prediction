import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sqlalchemy import create_engine 
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet 
from prophet.plot import plot_plotly, plot_components_plotly 

# Header/Title of report
pagetitle = "Gold Future Price Prediction Report"
st.set_page_config(page_title=pagetitle, layout="wide")
# Create a container for the title
title_container = st.container()
# Add HTML Title with CSS Styling
with title_container:
    title_html = """
    <h1 style="text-align: center;">""" + pagetitle + """</h1>
    """
    st.markdown(title_html, unsafe_allow_html=True)

# body of the report
# Loading the data
df = pd.read_csv("./gold.csv")

# Report Creation
df1 = df[["Date", "Volume"]]
df1.columns = ["ds", "y"]
st.write("This report contains the analysis on Gold Future prices dataset (GC00: Gold Continuous Contract Futures, USD). Below lies the Gold Future Prices chart spanning from the 22nd of April 2014 - 19th of January 2024")
st.write("There doesn't appear to be a consistent upward or downward trend over the entire time period. The price fluctuate erratically around a certain level.")
fig = px.line(df1, x="ds", y="y")
fig.update_layout(
    width=1300,  # Adjust width as needed
    height=500   # Adjust height as needed
)
st.plotly_chart(fig)


st.write("The historical data shows significant fluctuations in the gold prices, with notable peaks around 2019 and 2020. The forecast indicates a decline in Gold Continuous Contract for the remainder of 2024. This downward trend is primarily driven by previous economic indicators, interest rates, global events, market sentiments.")
# Define the training date range
start_date = '2014-01-01'
end_date = '2023-12-31'
start_date1 = '2024-01-01'
end_date1 = '2024-01-20'

# Filter the DataFrame to select only the training and testing data
train = df1[(df1['ds'] >= start_date) & (df1['ds'] <= end_date)]
test = df1[(df1['ds'] >= start_date1) & (df1['ds'] <= end_date1)]
m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods = 365) #MS - Monthly, H - Hourly
forecast = m.predict(future) 

# plotting the forecasted data
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df1['ds'],  # Use the original data for actual points
        y=df1['y'],
        mode='markers',  # Display as markers
        name='Actual Data'
    )
)
# Add the forecast line
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='black')
))
# Add the confidence interval (Bollinger Bands)
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    fill='tonexty',
    fillcolor='rgba(173, 216, 230, 0.5)',  # Light blue fill color
    line=dict(width=0),
    hoverinfo="skip",
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    fill='tonexty',
    fillcolor='rgba(173, 216, 230, 0.5)',  # Light blue fill color
    line=dict(width=0),
    hoverinfo="skip",
    showlegend=False
))
# Customize the layout
fig.update_layout(
    title='Forecast',
    xaxis_title='Date',
    yaxis_title='Value',
    width=1400,  # Adjust width as needed
    height=700   # Adjust height as needed
)
st.plotly_chart(fig)

st.write("**Recommendation**")
st.write("After accurately predicting a decline in gold prices, investors should reduce their exposure to gold-related assets, potentially mitigating losses during the downturn.")