# Importing necessary libraries
import yfinance as yf  # For fetching financial data
import pandas as pd  # For data manipulation
import numpy as np  # For numerical calculations
import scipy as sp  # For scientific computing
import matplotlib.pyplot as plt  # For plotting
from datetime import datetime  # For handling dates
from scipy.optimize import fsolve  # For solving equations
from scipy.stats import norm  # For normal distribution
import tkinter as tk  # For GUI
from tkinter import messagebox  # For displaying error messages

# Suppressing warnings related to iteration progress
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# Getting today's date
today = datetime.now()

def vol_nappe(sj, r, our_type, sigma, xtol, maxfev) :
    sj = sj  # Ticker symbol
    r = r  # Risk-free rate

    # Downloading historical data to get S0
    start_date = today.strftime('%Y-%m-%d')
    stock_data = yf.download(sj, start=start_date)
    S0 = stock_data['Close'].iloc[-1]

    option_data = yf.Ticker(sj)
    option_datas = option_data.options

    df_option_data = []

    # Iterating through option data for different expiration dates
    for expiration_date in option_datas:
        option_chain = option_data.option_chain(expiration_date)
        
        calls = option_chain.calls
        puts = option_chain.puts
        
        # Extracting call options data
        for option in calls.iterrows():
            strike = option[1]['strike']
            midprice = (option[1]['bid'] + option[1]['ask']) / 2
            df_option_data.append([expiration_date, strike, midprice, 'CALL'])

        # Extracting put options data
        for option in puts.iterrows():
            strike = option[1]['strike']
            midprice = (option[1]['bid'] + option[1]['ask']) / 2
            df_option_data.append([expiration_date, strike, midprice, 'PUT'])

    df_option_data = pd.DataFrame(df_option_data, columns=['expiration_date', 'strike', 'midprice', 'type'])
    df_option_data = df_option_data[(S0 * 0.8 < df_option_data['strike']) & (df_option_data['strike'] < S0 * 1.2)]

    df_option_data['days_to_expiry'] = df_option_data['expiration_date']
    df_option_data['days_to_expiry'] = pd.to_datetime(df_option_data['days_to_expiry'])
    
    df_option_data['expiration_date'] = (df_option_data['days_to_expiry'] - today).dt.days

    df_option_data = df_option_data[df_option_data['expiration_date'] > 0]
    df_option_data = df_option_data[df_option_data['expiration_date'] < 100]  # Filter expiration dates so they are not too far in the future
    df_option_data = df_option_data.set_index(['expiration_date', 'strike', 'type']).sort_index()

    df_option_data = df_option_data.pivot_table(index=['expiration_date', 'strike'], columns='type', values='midprice')

    # Black-Scholes-Merton (BSM) formula for option pricing
    def BSM(sigma, S0, K, P, T, r, type_) : 
        d1 = (np.log(S0/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if type_ == 'CALL' : 
            return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - P
        elif type_ == 'PUT' : 
            return K * np.exp(-r * T) * norm.cdf(-d2) - P - S0 * norm.cdf(-d1) 

    imp_vol_df = []
    # Calculating implied volatility using BSM formula
    for index, value in df_option_data[our_type].items():
        T = index[0] / 365
        K = index[1]
        sigma0 = sigma  # Initial value of sigma 
        imp_vol = fsolve(BSM, sigma0, args=(S0, K, value, T, r, our_type), xtol=xtol, maxfev=maxfev)
        imp_vol_scalar = float(imp_vol[0])
        imp_vol_df.append(imp_vol_scalar)

    imp_vol_df_indexed = pd.Series(imp_vol_df, index=df_option_data[our_type].index)

    df_interpolated = imp_vol_df_indexed.unstack(0).interpolate(method='linear')

    # Creating a 3D plot of the volatility surface
    x, y = np.meshgrid(np.array(df_option_data[our_type].index.levels[0]), np.array(df_option_data[our_type].index.get_level_values(1).unique()))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, np.array(df_interpolated), cmap='plasma')

    plt.xlabel("Maturity")
    plt.ylabel("Strike")
    ax.set_zlabel("Volatility")

    fig.colorbar(surf)

    plt.show()

def generate_volatility_surface():
    sous_jacent = sous_jacent_entry.get()
    r_value = float(r_entry.get())
    our_type_value = our_type_entry.get()
    sigma_value = float(sigma_entry.get())
    xtol_value = float(xtol_entry.get())
    maxfev_value = int(maxfev_entry.get())
    
    # Error handling for empty input fields and invalid option type
    if sous_jacent.strip() == "":
        messagebox.showerror("Error", "Please enter underlying asset.")
        return
    if our_type_value not in ['CALL', 'PUT']:
        messagebox.showerror("Error", "Invalid option type. Choose 'CALL' or 'PUT'.")
        return
    vol_nappe(sous_jacent, r_value, our_type_value, sigma_value, xtol_value, maxfev_value)

# Creating the main window
root = tk.Tk()
root.title("Volatility Surface")

# Creating widgets
sous_jacent_label = tk.Label(root, text="Underlying Asset:")
sous_jacent_entry = tk.Entry(root)

r_label = tk.Label(root, text="Risk-Free Rate:")
r_entry = tk.Entry(root)

our_type_label = tk.Label(root, text="Option Type (CALL/PUT):")
our_type_entry = tk.Entry(root)

sigma_label = tk.Label(root, text="Sigma:")
sigma_entry = tk.Entry(root)

xtol_label = tk.Label(root, text="Tolerance (xtol):")
xtol_entry = tk.Entry(root)

maxfev_label = tk.Label(root, text="Maximum Number of Iterations (maxfev):")
maxfev_entry = tk.Entry(root)

generate_button = tk.Button(root, text="Generate", command=generate_volatility_surface)

# Placing widgets in the window
sous_jacent_label.grid(row=0, column=0, padx=10, pady=10)
sous_jacent_entry.grid(row=0, column=1, padx=10, pady=10)

r_label.grid(row=1, column=0, padx=10, pady=10)
r_entry.grid(row=1, column=1, padx=10, pady=10)

our_type_label.grid(row=2, column=0, padx=10, pady=10)
our_type_entry.grid(row=2, column=1, padx=10, pady=10)

sigma_label.grid(row=3, column=0, padx=10, pady=10)
sigma_entry.grid(row=3, column=1, padx=10, pady=10)

xtol_label.grid(row=4, column=0, padx=10, pady=10)
xtol_entry.grid(row=4, column=1, padx=10, pady=10)

maxfev_label.grid(row=5, column=0, padx=10, pady=10)
maxfev_entry.grid(row=5, column=1, padx=10, pady=10)

generate_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

# Running the main loop
root.mainloop()
