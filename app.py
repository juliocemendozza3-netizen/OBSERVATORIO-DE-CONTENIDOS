import streamlit as st
import pandas as pd

st.title("📊 Observatorio CRC")

df = pd.read_csv("observatorio.csv")

st.dataframe(df)
