import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta

def bates_model_log_likelihood(params, returns):
    mu, kappa, theta, sigma, jump_mean, jump_std, jump_intensity = params
    dt = 1/365  # daily data, annualized
    
    lambda_adj = jump_mean * jump_intensity
    var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt)) + \
          jump_intensity * dt * (jump_mean**2 + jump_std**2)
    
    mean = (mu - lambda_adj - 0.5 * sigma**2) * dt + \
           (1 - np.exp(-kappa * dt)) * (theta - (mu - lambda_adj) / kappa)
    
    log_likelihood = np.sum(-0.5 * np.log(2 * np.pi * var) - 
                            (returns - mean)**2 / (2 * var))
    return -log_likelihood

def fit_bates_model(returns):
    initial_guess = [0.1, 2.0, 0.1, 0.3, 0.0, 0.1, 10.0]
    bounds = [(None, None), (0, None), (None, None), (0, None), 
              (None, None), (0, None), (0, None)]
    
    result = minimize(bates_model_log_likelihood, initial_guess, 
                      args=(returns,), bounds=bounds, method='L-BFGS-B')
    
    return result.x

# Fetch Ethereum price data from CoinGecko
cg = CoinGeckoAPI()
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
ethereum_data = cg.get_coin_market_chart_range_by_id(
    id='ethereum',
    vs_currency='usd',
    from_timestamp=int(start_date.timestamp()),
    to_timestamp=int(end_date.timestamp())
)

# Process the data
prices = pd.DataFrame(ethereum_data['prices'], columns=['timestamp', 'price'])
prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
prices = prices.set_index('date')
prices = prices.sort_index()

# Calculate log returns
log_returns = np.log(prices['price'] / prices['price'].shift(1)).dropna()

# Fit Bates model
bates_params = fit_bates_model(log_returns.values)

# Print results
param_names = ['mu', 'kappa', 'theta', 'sigma', 'jump_mean', 'jump_std', 'jump_intensity']
for name, value in zip(param_names, bates_params):
    print(f"{name}: {value:.6f}")

# Calculate annualized volatility
total_variance = (bates_params[3]**2 / (2 * bates_params[1])) + \
                 bates_params[6] * (bates_params[4]**2 + bates_params[5]**2)
annualized_volatility = np.sqrt(total_variance) * np.sqrt(365)
print(f"Annualized volatility: {annualized_volatility:.6f}")