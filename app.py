
from pathlib import Path

import streamlit as st
import pandas as pd

st.title("Student Performance Analysis")

df = pd.read_csv(Path(__file__).with_name("student_performance.csv"))

subjects = ["Maths", "Science", "English"]
df["Average_Marks"] = df[subjects].mean(axis=1)

st.metric("Total Students", len(df))
st.metric("Overall Average Marks", round(df[subjects].to_numpy().mean(), 2))

st.subheader("Student Data")
st.dataframe(df)

st.subheader("Average Marks by Subject")
st.bar_chart(df[subjects].mean())

st.subheader("Student Attendance")
st.bar_chart(df.set_index("Name")["Attendance"])
