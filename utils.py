import numpy as np 
import pandas as pd

from prophet import Prophet
from alibi_detect.od import SpectralResidual,IForest
import numpy as np
import warnings
import matplotlib.pyplot as plt


df = pd.read_csv("datasets/sensor.csv", parse_dates=["timestamp"], index_col= "timestamp" )
df = df.drop("Unnamed: 0", axis=1)
print("Read dataset...")

# Prophet functions and plotting

def plot_sensors(sensor_name):
    #initializing prophet object
    test = Prophet()

    # Creating dataframe for prophet training
    temp_df = pd.DataFrame()
    temp_df["ds"] = df.index
    temp_df["y"] = df[sensor_name].values

    print("Fitting model...")

    # Fitting model with 25000 values
    test.fit(temp_df[:50000])

    print("Making predictions...")

    # Making predictions for 1 day
    future = test.make_future_dataframe(periods=1)
    forecast = test.predict(future)

    # Adding target values to the forecasted dataframe
    forecast["orig_anomaly"] = df["machine_status"][:50001].values

    print("Plotting predictions...")

    # Plotting the forecasted model
    import matplotlib.pyplot as plt
    figsize=(25, 6)
    xlabel='ds'
    ylabel='y'
    fig = plt.figure(facecolor='w', figsize=figsize)

    ax = fig.add_subplot(111)
    fcst_t = forecast['ds'].dt.to_pydatetime()
    ax.plot(test.history['ds'].dt.to_pydatetime(), test.history['y'], 'k.', color="yellow",label="orginal sensor data", alpha=0.8)

    ax.plot(fcst_t, forecast['yhat'], ls='-', c='#0072B2', label="Predicted values")



    ax.fill_between(fcst_t, forecast['yhat_lower'], forecast['yhat_upper'], color='#0072B2', alpha=0.2, label = "Uncertainity area")

    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

    ax.scatter(forecast[forecast["orig_anomaly"] == "BROKEN"]["ds"], forecast[forecast["orig_anomaly"] == "BROKEN"]["trend"], color='black', marker='D', zorder=2, label='BROKEN')

    ax.scatter(forecast[forecast["orig_anomaly"] == "RECOVERING"]["ds"], forecast[forecast["orig_anomaly"] == "RECOVERING"]["trend"], color='green', marker='x', zorder=2, label='RECOVERING')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()




    plt.legend()

# print("Reading sensor data..")
# plot_sensors("sensor_02")

# Alibi_detect functions and plotting
def alibi_anomaly_detection( sensor_name: str, method_name=None):
    sensor_name = sensor_name
    df = pd.read_csv("datasets/sensor.csv", parse_dates=["timestamp"], index_col= "timestamp" )
    df = df.drop("Unnamed: 0", axis=1)
    
    test_df = pd.DataFrame()
    test_df["time"] = df[:100000].index
    test_df["values"] = df[sensor_name][:100000].values
    
    if method_name is None:
        od = SpectralResidual(
                threshold=1.,
                window_amp=20,
                window_local=20,
                n_est_points=10,
                n_grad_points=5
        )
        print("Executing Spectral Residual method... ")
    else:
        od = IForest(threshold=None,  # threshold for outlier score
                    n_estimators=100)
        od.fit(np.array(test_df["values"].dropna()).reshape(-1,1))
        print("Executing IForest method... ")
    
    scores = od.score(np.array(df[sensor_name].dropna().values).reshape(-1,1))

    if np.isnan(scores).sum() != 0:
        warnings.warn("Scores are None. Check alibi_anomaly_detection function")
    

    # Plotting data
    df1 = pd.read_csv("datasets/sensor.csv")
    anomalies_index = df1[df1["machine_status"]=="BROKEN"][sensor_name].index
    anomalies_values = df1[df1["machine_status"]=="BROKEN"][sensor_name].values
    
    
    plt.plot(df[sensor_name].values, label="Original data")
    plt.plot(scores, label="Anomaly Scores")
    plt.scatter(x=anomalies_index, y=anomalies_values, color="black", label = "Original Status: BROKEN")

    print("Plotting finished")

# alibi_anomaly_detection(sensor_name="sensor_02", method_name="IForest")