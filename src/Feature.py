import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:\Users\vansh\OneDrive\Desktop\PROJECTS\AIML\SUPERVISED\LINEAR\CRYPTO_PRICE_PREDICTOR USING LINEAR REGRESSION\data\btcusd_1-min_data.csv')
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Date', inplace=True)
df.drop('Timestamp', axis=1, inplace=True)
df_daily = df.resample('D').agg({
    'Open':'first', 'High':'max',
    'Low':'min', 'Close':'last', 'Volume':'sum'
})
df_daily.dropna(inplace=True)



df_daily['Target'] = df_daily['Close'].shift(-1)

df_daily['Price_Change'] = df_daily['Close'] - df_daily['Open']

df_daily['High_Low_Range'] = df_daily['High'] - df_daily['Low']

df_daily.dropna(inplace=True)

print("Features ready. Shape:", df_daily.shape)
print(df_daily[['Close', 'Target', 'Price_Change', 'High_Low_Range']].head())


X = df_daily[['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'High_Low_Range']]
y = df_daily['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows :", X_test.shape[0])

model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained.")

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nRMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")


plt.figure(figsize=(12, 5))
plt.plot(y_test.values, color='blue', label='Actual BTC Price')
plt.plot(y_pred,        color='red',  label='Predicted BTC Price', alpha=0.7)
plt.title('Bitcoin Price — Actual vs Predicted')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('../data/btc_prediction.png')
print("\nChart saved to data/btc_prediction.png")

print("\n── Model Insights ──────────────────────────────")
print(f"Trained on : {X_train.shape[0]} days of BTC data")
print(f"Tested on  : {X_test.shape[0]} days of BTC data")
print(f"RMSE       : ${rmse:,.2f} average error per prediction")
print(f"R² Score   : {r2:.4f} (1.0 = perfect)")

last_row = X.iloc[-1].values.reshape(1, -1)
tomorrow = model.predict(last_row)[0]
print(f"\nPredicted next BTC price: ${tomorrow:,.2f}")