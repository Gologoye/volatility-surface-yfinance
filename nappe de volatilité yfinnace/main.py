import yfinance as yf
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import fsolve
from scipy.stats import norm


import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

today = datetime.now()


sj = 'X' # sous jacent 
r = 0.05  # Taux sans risque

# Télécharger les données historiques pour obtenir S0
stock_data = yf.download(sj, start='2023-01-01', end='2024-04-10')
S0 = stock_data['Close'].iloc[-1]

option_data = yf.Ticker(sj)
option_datas = option_data.options

df_option_data = []

for expiration_date in option_datas:
    option_chain = option_data.option_chain(expiration_date)
    
    calls = option_chain.calls
    puts = option_chain.puts
    
    for option in calls.iterrows():
        strike = option[1]['strike']
        midprice = (option[1]['bid'] + option[1]['ask']) / 2
        df_option_data.append([expiration_date, strike, midprice, 'CALL'])

    for option in puts.iterrows():
        strike = option[1]['strike']
        midprice = (option[1]['bid'] + option[1]['ask']) / 2
        df_option_data.append([expiration_date, strike, midprice, 'PUT'])

df_option_data = pd.DataFrame(df_option_data, columns=['expiration_date', 'strike', 'midprice', 'type'])
df_option_data = df_option_data[(S0 * 0.8 < df_option_data['strike']) & (df_option_data['strike'] < S0 * 1.2)]

df_option_data['days_to_expiry'] = df_option_data['expiration_date']
df_option_data['days_to_expiry'] = pd.to_datetime(df_option_data['days_to_expiry'])
df_option_data['expiration_date'] = (df_option_data['days_to_expiry'] - today).dt.days

df_option_data = df_option_data[df_option_data['expiration_date'] < 100]  # Filter expiration dates pour que l'expiration ne soient pas trop lointaine
df_option_data = df_option_data.set_index(['expiration_date', 'strike', 'type']).sort_index()

df_option_data = df_option_data.pivot_table(index=['expiration_date', 'strike'], columns='type', values='midprice')

def BSM(sigma, S0, K, P, T, r, type_) : 
    d1 = (np.log(S0/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if type_ == 'CALL' : 
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - P
    elif type_ == 'PUT' : 
        return K * np.exp(-r * T) * norm.cdf(-d2) - P - S0 * norm.cdf(-d1) 

our_type = 'CALL'

imp_vol_df = []
for index, value in df_option_data[our_type].items():
    T = index[0] / 365
    K = index[1]
    sigma0 = 0.60 # valeur de sigma 
    imp_vol = fsolve(BSM, sigma0, args=(S0, K, value, T, r, our_type), xtol=1e-8, maxfev=500)   # parametres d'aproximation de la volatilité (xtol et maxfev)
    imp_vol_scalar = float(imp_vol[0])
    imp_vol_df.append(imp_vol_scalar)




#Matplotlib Smile de volatilité



maturities = df_option_data[our_type].index.levels[0]
day1 = df_option_data[our_type].index.levels[0][0]
day2 = df_option_data[our_type].index.levels[0][3]
day3 = df_option_data[our_type].index.levels[0][6]
day4 = df_option_data[our_type].index.levels[0][7]

imp_vol_df_indexed = pd.Series(imp_vol_df, index=df_option_data[our_type].index)

plt.figure(figsize=(12, 8))
plt.title('Implied volatility smile')

# Utilisez imp_vol_df pour tracer les données
plt.plot(pd.Series(imp_vol_df, index=df_option_data[our_type].index).loc[day1], label=f'maturity_{day1}_days')
plt.plot(pd.Series(imp_vol_df, index=df_option_data[our_type].index).loc[day2], label=f'maturity_{day2}_days')
plt.plot(pd.Series(imp_vol_df, index=df_option_data[our_type].index).loc[day3], label=f'maturity_{day3}_days')
plt.plot(pd.Series(imp_vol_df, index=df_option_data[our_type].index).loc[day4], label=f'maturity_{day4}_days')

plt.legend()



#Nappe de volatilité

# imp_vol_df_indexed = pd.Series(imp_vol_df, index=df_option_data[our_type].index)


df_interpolated = imp_vol_df_indexed.unstack(0).interpolate(method='linear')


x, y = np.meshgrid(np.array(df_option_data[our_type].index.levels[0]), np.array(df_option_data[our_type].index.get_level_values(1).unique()))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, np.array(df_interpolated), cmap='plasma')

plt.xlabel("Maturity")
plt.ylabel("Strike")
ax.set_zlabel("Volatility")

fig.colorbar(surf)

plt.show()

