import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng()


def simulation_anomaly(drift_start, drift_end, moy_power, noise, nb_points):
    
    """
   Simulates power measurements of a nuclear reactor with anomalies.
   
   Parameters:
   - drift_start : start value of power drift (MW)
   - drift_end   : end value of power drift (MW)
   - moy_power   : normal power level (MW)
   - noise       : gaussian noise standard deviation (MW)
   - nb_points   : number of measurements
   """
    
    #Creatiion of a noisy measure of power
    power_drift = np.linspace(drift_start, drift_end, 1000)
    normal_power = rng.normal(moy_power, noise, nb_points)
    noisy_power = np.concatenate([normal_power[0:6500],
                                  normal_power[6500:7500] + power_drift,
                                  normal_power[7500:10000]
])
    #Add manualy some anomalies
    anomalies_under = rng.uniform(-500, 0, size=10)
    anomalies_above = rng.normal(2000, 50, 10)
    indices_1 = rng.integers(0, 10000, size=10)
    noisy_power[indices_1] = anomalies_under
    indices_2 = rng.integers(0, 10000, size=10)
    noisy_power[indices_2] = anomalies_above
    noisy_power = noisy_power.reshape(-1, 1)
    time_s = np.linspace(1, 10000, 10000)

    #Creation of a model to identify the anomaly
    model = IsolationForest()
    model.fit(noisy_power)
    prediction = model.predict(noisy_power)

    #Graph with the normal measure and the anomaly one
    plt.style.use('dark_background')
    plt.scatter(time_s[prediction == 1], 
                noisy_power[prediction == 1], s=0.2, c='g', label='Normal')
    plt.scatter(time_s[prediction == -1], 
                noisy_power[prediction == -1], s=2, c='r', label='Anomaly')
    plt.xlabel('Time_s')
    plt.ylabel('Power in MW')
    plt.title("Noisy measure of power simulated for a nuclear reactor", fontsize=15)
    plt.legend(markerscale=5)
    plt.show()

    nb_anomaly = (prediction == -1).sum()
    anomaly_percentage = (nb_anomaly / time_s.size) * 100
    print('Total number of anomaly = ', nb_anomaly)
    print('Percentage of times in anomaly =', anomaly_percentage.round(4),'%')
    
    return noisy_power, prediction



#test
simulation_anomaly(1, 1000, 1000, 50, 10000)
