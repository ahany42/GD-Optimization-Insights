import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r'D:\ASU\term 6\Machine Learning\Assignment\assignment1dataset.csv')
#Data Inspection
# print(df.head(5))
# print(df.info())
# print(df.describe())
# print("NaN Count",df.isna().sum())
# print("Duplicate Count",df.duplicated().sum())
# print("Null Count",df.isnull().sum())


scaler = MinMaxScaler(feature_range=(-1,1))
scaled_features = scaler.fit_transform(df[['RevenuePerDay', 'NCustomersPerDay', 'AverageOrderValue','WorkingHoursPerDay', 'NEmployees', 'MarketingSpendPerDay','LocationFootTraffic']])
df_scaled = pd.DataFrame(scaled_features, columns=['RevenuePerDay', 'NCustomersPerDay', 'AverageOrderValue','WorkingHoursPerDay', 'NEmployees', 'MarketingSpendPerDay', 'LocationFootTraffic'])
def linear_regression(learning_rate_list, epoch_list, is_alpha):
    features = ['NCustomersPerDay', 'AverageOrderValue', 'WorkingHoursPerDay',  'NEmployees', 'MarketingSpendPerDay', 'LocationFootTraffic']
    Y = df_scaled['RevenuePerDay']  
    
    mse_map = {feature: [] for feature in features}  
    for feature in features:
        X=df_scaled[feature]
        n = float(len(X)) 
        for param in learning_rate_list if is_alpha else epoch_list:
            L = param if is_alpha else learning_rate_list[0]  
            epochs = param if not is_alpha else epoch_list[0]  
            m=0
            c=0
            for _ in range(epochs):
                y_pred = m * X + c
                dm = np.sum((y_pred-Y) * X)/n
                dc = np.sum(y_pred-Y)/n
                m = m - L * dm
                c = c- L * dc
            mse = metrics.mean_squared_error(Y, y_pred)
            mse_map[feature].append(mse)
            
    plt.figure(figsize=(10, 6))
    
    for feature in features:
        plt.plot(learning_rate_list if is_alpha else epoch_list, mse_map[feature], marker='o', label=feature)
     
    plt.xlabel("Learning Rate (alpha)" if is_alpha else "Epochs", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.title("MSE vs Learning Rate" if is_alpha else "MSE vs Epochs", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    if(is_alpha):
        print("\n","MSE vs Learning Rate")
        for key, value in mse_map.items():
            print(f"{key}: {value}")
        print("\n")
    else:
        print("MSE vs Epochs")
        for key, value in mse_map.items():
            print(f"{key}: {value}")

linear_regression([0.0001, 0.001, 0.01, 0.1, 0.3, 0.9], [100], True)  
linear_regression([0.001], [100,200,1000], False) 
