#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 15:11:52 2025

@author: damianesene
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import numpy as np
import pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer import Pitch, VerticalPitch
import time
import seaborn as sns
from scipy.stats import zscore 

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load Data
file_path = "/Users/damianesene/Downloads/EPL_Results.ods"

results_table = pd.read_excel(file_path)

# Clean data

results_table = results_table.dropna(subset=["FTHG", "FTAG", "FTR"])

results_table = results_table.reset_index(drop=True)

results_table["Date"] = pd.to_datetime(results_table["Date"])


# Testing with target variable Home win
results_table["home_win"] = (results_table["FTR"] == "H").astype(int)

features = [
    "AvgH", "AvgD", "AvgA",   # market strength
    "HS", "AS",
    "HST", "AST",
    "HC", "AC"
]

results_table = results_table.dropna(subset=features)

# Prepare data and train the model
X = results_table[features]
y = results_table["home_win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))




