import numpy as np 
import pandas as pd

from prophet import Prophet

# Prophet functions and plotting


model = Prophet()

df = pd.read_csv("/dataset/sensor.csv", parse_dates=["timestamp"], index_col= "timestamp" )
df.head()

# Alibi_detect functions and plotting
