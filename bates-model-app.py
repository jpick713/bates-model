import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Function to fetch crypto data
@st.cache_data
def fetch_crypto_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# Function to calculate returns
def calculate_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

# Function to calculate realized variance
def calculate_realized_variance(returns, window=30, min_periods=10):
    return returns.ewm(span=window, min_periods=min_periods).var()

# Function to calculate exponentially weighted correlation
def exponential_weighted_correlation(data, span=30):
    ewm = data.ewm(span=span)
    means = ewm.mean()
    variances = ewm.var()
    stds = np.sqrt(variances)
    corr = ewm.cov() / (stds.values.reshape((-1, 1)) @ stds.values.reshape((1, -1)))
    return corr.iloc[-1]

# Function to estimate Bates model parameters
def estimate_bates_parameters(returns, realized_variance):
    dt = 1/252  # Assuming daily data
    
    def objective(params):
        kappa, theta, sigma, lambda_jump, mu_jump, sigma_jump = params
        
        # Theoretical moments
        mean_theory = (theta - realized_variance.iloc[0]) * (1 - np.exp(-kappa * dt)) + lambda_jump * mu_jump * dt
        var_theory = (sigma**2 / (2*kappa)) * (1 - np.exp(-2*kappa*dt)) + lambda_jump * (mu_jump**2 + sigma_jump**2) * dt
        skew_theory = (lambda_jump * dt * (mu_jump**3 + 3*mu_jump*sigma_jump**2)) / (var_theory**(3/2))
        kurt_theory = (lambda_jump * dt * (mu_jump**4 + 6*mu_jump**2*sigma_jump**2 + 3*sigma_jump**4)) / (var_theory**2)
        
        # Empirical moments
        mean_emp = returns.mean()
        var_emp = returns.var()
        skew_emp = returns.skew()
        kurt_emp = returns.kurtosis()
        
        # Squared errors
        errors = [
            (mean_theory - mean_emp)**2,
            (var_theory - var_emp)**2,
            (skew_theory - skew_emp)**2,
            (kurt_theory - kurt_emp)**2
        ]
        
        return sum(errors)
    
    # Initial guess and bounds
    initial_guess = [2.0, realized_variance.mean(), 0.3, 1.0, -0.05, 0.1]
    bounds = [(0, 10), (0, 1), (0, 1), (0, 10), (-0.5, 0.5), (0, 1)]
    
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    return result.x

# Streamlit app
st.title('Multi-Asset Bates Model for Cryptocurrencies')

# Sidebar for user inputs
st.sidebar.header('Input Parameters')
tickers = st.sidebar.multiselect('Select Cryptocurrencies', ['BTC-USD', 'ETH-USD', 'LINK-USD', 'ADA-USD', 'DOT-USD', 'XRP-USD'], default=['BTC-USD', 'ETH-USD', 'LINK-USD'])
days = st.sidebar.slider('Number of days for historical data', 30, 365, 30)
simulation_days = st.sidebar.slider('Number of days to simulate', 1, 365, 30)
num_simulations = st.sidebar.slider('Number of simulations', 100, 10000, 1000)

# Fetch data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
prices = fetch_crypto_data(tickers, start_date, end_date)

# Calculate returns and realized variance
returns = calculate_returns(prices)
realized_variance = calculate_realized_variance(returns)

# Combine returns and realized variances
combined_data = pd.concat([returns, realized_variance], axis=1)
combined_data.columns = [f'{ticker}_return' for ticker in tickers] + [f'{ticker}_variance' for ticker in tickers]

# Calculate the full correlation matrix
full_corr_matrix = exponential_weighted_correlation(combined_data)

# Estimate Bates model parameters
params = {}
for ticker in tickers:
    params[ticker] = estimate_bates_parameters(returns[ticker], realized_variance[ticker])

# Display correlation matrix
st.header('Correlation Matrix')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(full_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
plt.title('Full Correlation Matrix (Including Cross-Correlations)')
st.pyplot(fig)

# Display estimated parameters
st.header('Estimated Bates Model Parameters')
for ticker in tickers:
    st.subheader(ticker)
    kappa, theta, sigma, lambda_jump, mu_jump, sigma_jump = params[ticker]
    st.write(f'Mean Reversion Speed (kappa): {kappa:.4f}')
    st.write(f'Long-term Variance (theta): {theta:.6f}')
    st.write(f'Volatility of Variance (sigma): {sigma:.4f}')
    st.write(f'Jump Intensity (lambda): {lambda_jump:.4f}')
    st.write(f'Mean Jump Size (mu_jump): {mu_jump:.4f}')
    st.write(f'Jump Size Volatility (sigma_jump): {sigma_jump:.4f}')

# Function for multi-asset Bates model simulation
def multi_asset_bates_simulation(S0, v0, r, T, params, corr_matrix, num_steps, num_paths, num_assets):
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    
    L = np.linalg.cholesky(corr_matrix)
    
    S = np.zeros((num_paths, num_steps + 1, num_assets))
    v = np.zeros((num_paths, num_steps + 1, num_assets))
    S[:, 0] = S0
    v[:, 0] = v0
    
    for i in range(1, num_steps + 1):
        Z = np.random.normal(0, 1, (num_paths, 2 * num_assets))
        W = np.dot(Z, L.T)
        
        dW1 = W[:, :num_assets]
        dW2 = W[:, num_assets:]
        
        for j in range(num_assets):
            kappa, theta, sigma, lambda_jump, mu_jump, sigma_jump = params[j]
            
            dN = np.random.poisson(lambda_jump * dt, num_paths)
            J = np.random.normal(mu_jump, sigma_jump, num_paths) * dN
            
            v[:, i, j] = np.maximum(v[:, i-1, j] + kappa * (theta - v[:, i-1, j]) * dt + 
                                    sigma * np.sqrt(v[:, i-1, j]) * dW2[:, j] * sqrt_dt, 0)
            
            S[:, i, j] = S[:, i-1, j] * np.exp((r - 0.5 * v[:, i-1, j] - lambda_jump * (np.exp(mu_jump + 0.5 * sigma_jump**2) - 1)) * dt + 
                                               np.sqrt(v[:, i-1, j]) * dW1[:, j] * sqrt_dt + J)
    
    return S

# Run simulation
S0 = prices.iloc[-1].values
v0 = realized_variance.iloc[-1].values
r = 0.05  # Risk-free rate
T = simulation_days / 252  # Time horizon in years
num_steps = simulation_days
num_assets = len(tickers)

sim_params = [params[ticker] for ticker in tickers]
S = multi_asset_bates_simulation(S0, v0, r, T, sim_params, full_corr_matrix.values, num_steps, num_simulations, num_assets)

# Plot simulation results
st.header('Simulation Results')
fig, axs = plt.subplots(num_assets, 1, figsize=(12, 5*num_assets), sharex=True)
for i, ticker in enumerate(tickers):
    axs[i].plot(S[:, :, i].T, alpha=0.1, color='blue')
    axs[i].plot(S[:, :, i].mean(axis=0), color='red', linewidth=2)
    axs[i].set_title(f'{ticker} Price Paths')
    axs[i].set_ylabel('Price')
axs[-1].set_xlabel('Days')
plt.tight_layout()
st.pyplot(fig)

# Calculate and display risk metrics
st.header('Risk Metrics')
for i, ticker in enumerate(tickers):
    returns_sim = np.log(S[:, -1, i] / S0[i])
    var_95 = np.percentile(returns_sim, 5)
    var_99 = np.percentile(returns_sim, 1)
    es_95 = returns_sim[returns_sim <= var_95].mean()
    
    st.subheader(ticker)
    st.write(f'95% VaR: {-var_95:.2%}')
    st.write(f'99% VaR: {-var_99:.2%}')
    st.write(f'95% Expected Shortfall: {-es_95:.2%}')

st.write("""
Note: This app provides a simplified implementation of the multi-asset Bates model for educational purposes. 
Real-world applications would require more sophisticated parameter estimation techniques and additional model validation.
""")
