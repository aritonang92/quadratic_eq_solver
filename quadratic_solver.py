# -*- coding: utf-8 -*-
"""Untitled10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f1BHauP34tOjWKi32dLvlfk7UHah8S6K
"""

import pandas as pd
import streamlit as st

st.title("A Simple Quadratic Equation Solver")

name = st.text_input("Enter your name"," ")

st.write(f"Hello {name}!")

x = st.number_input("input your x:")
y = st.number_input("input your y:")

df = x**2 - 2-y + 17

st.write("hasilnya adalah: {df}")

