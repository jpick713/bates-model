import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
from numpy import linalg as LA

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
    # Calculate daily variance
    daily_variance = returns.ewm(span=window, min_periods=min_periods).var()
    # Annualize the variance
    annualized_variance = daily_variance * 365
    return annualized_variance

# Function to calculate exponentially weighted correlation
def exponential_weighted_correlation(data, span=30):
    ewm = data.ewm(span=span)
    means = ewm.mean()
    centered = data - means
    cov = centered.ewm(span=span).cov()
    
    # Extract the last timestamp for all pairs
    last_timestamp = cov.index.get_level_values(0)[-1]
    corr = cov.loc[last_timestamp].corr()
    
    return corr

# Function to estimate Bates model parameters
def estimate_bates_parameters(returns, realized_variance):
    dt = 1/365  # Assuming daily data for crypto (365 days per year)
    
    def objective(params):
        kappa, theta, sigma, lambda_jump, mu_jump, sigma_jump = params
        
        # Theoretical moments (using annualized parameters)
        mean_theory = (theta - realized_variance.iloc[0]) * (1 - np.exp(-kappa * dt)) + lambda_jump * mu_jump * dt
        var_theory = (sigma**2 / (2*kappa)) * (1 - np.exp(-2*kappa*dt)) + lambda_jump * (mu_jump**2 + sigma_jump**2) * dt
        skew_theory = (lambda_jump * dt * (mu_jump**3 + 3*mu_jump*sigma_jump**2)) / (var_theory**(3/2))
        kurt_theory = (lambda_jump * dt * (mu_jump**4 + 6*mu_jump**2*sigma_jump**2 + 3*sigma_jump**4)) / (var_theory**2)
        
        # Empirical moments (annualized)
        mean_emp = returns.mean() * 365
        var_emp = returns.var() * 365
        skew_emp = returns.skew() / np.sqrt(365)
        kurt_emp = returns.kurtosis() / 365
        
        # Squared errors
        errors = [
            (mean_theory - mean_emp)**2,
            (var_theory - var_emp)**2,
            (skew_theory - skew_emp)**2,
            (kurt_theory - kurt_emp)**2
        ]
        
        return sum(errors)
    
    # Initial guess and bounds (using annualized values)
    initial_guess = [2.0, realized_variance.iloc[-1], 0.3, 1.0, 0.0, 0.1]
    bounds = [(0, 10), (0, 1), (0, 1), (0, 10), (-0.5, 0.5), (0, 1)]
    
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    return result.x

def calculate_worst_of_payoff(S, K):
    """Calculate the payoff of a worst-of put option"""
    worst_performance = np.min(S[:, -1, :] / S[:, 0, :], axis=1)
    return np.maximum(K - worst_performance * K, 0)

def calculate_knockin_worst_of_payoff(S, K, knock_in_barrier):
    """Calculate the payoff of a knock-in worst-of put option"""
    worst_performance = np.min(S[:, -1, :] / S[:, 0, :], axis=1)
    knocked_in = np.any(np.min(S / S[:, 0, :][:, np.newaxis, :], axis=2) <= knock_in_barrier, axis=1)
    return np.maximum(K - worst_performance, 0) * knocked_in

def calculate_premium_and_apy(payoffs, notional, r, T):
    """Calculate the premium and APY"""
    mean_payoff = np.mean(payoffs)
    premium_pct = mean_payoff * 100
    apy = (np.exp(np.log(1 + premium_pct / 100) / T) - 1) * 100
    return premium_pct, apy


st.title('Multi-Asset Bates Model for Cryptocurrencies: Knock-In Worst-of Option')

# Sidebar for user inputs
st.sidebar.header('Input Parameters')
tickers = st.sidebar.multiselect('Select Cryptocurrencies', ['BTC-USD', 'ETH-USD', 'LINK-USD', 'ADA-USD', 'DOT-USD', 'XRP-USD'], default=['BTC-USD', 'ETH-USD', 'LINK-USD'])
days = st.sidebar.slider('Number of days for historical data', 30, 365, 30)
simulation_days = st.sidebar.slider('Number of days to simulate', 1, 365, 30)
num_simulations = st.sidebar.slider('Number of simulations', 1000, 50000, 10000)
notional = st.sidebar.number_input('Notional Amount ($)', min_value=1000, value=10000, step=1000)
strike_pct = st.sidebar.slider('Strike Price (% of initial price)', 80, 120, 100, 1)
knock_in_pct = st.sidebar.select_slider('Knock-In Barrier (% of initial price)', options=range(10, 95, 5), value=80)
vol_cap_pct = st.sidebar.slider('Initial Volatility Cap (%)', 100, 150, 100, 10)

# Fetch data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
prices = fetch_crypto_data(tickers, start_date, end_date)

# Calculate returns and realized variance
returns = calculate_returns(prices)
realized_variance = calculate_realized_variance(returns)

# Apply volatility cap
vol_cap = (vol_cap_pct / 100) ** 2  # Convert percentage to variance
realized_variance = realized_variance.clip(upper=vol_cap)

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

def nearest_positive_definite(A):
    """
    Find the nearest positive-definite matrix to input A
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = LA.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    
    if is_positive_definite(A3):
        return A3
    
    spacing = np.spacing(LA.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(LA.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    
    return A3

def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = LA.cholesky(B)
        return True
    except LA.LinAlgError:
        return False

# Multi_asset_bates_simulation function
def multi_asset_bates_simulation(S0, v0, r, T, params, corr_matrix, num_steps, num_paths, num_assets):
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    
    # Ensure the correlation matrix is positive definite
    corr_matrix_pd = nearest_positive_definite(corr_matrix)
    L = np.linalg.cholesky(corr_matrix_pd)
    
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
T = simulation_days / 365  # Time horizon in years
num_steps = simulation_days
num_assets = len(tickers)

sim_params = [params[ticker] for ticker in tickers]
S = multi_asset_bates_simulation(S0, v0, r, T, sim_params, full_corr_matrix.values, num_steps, num_simulations, num_assets)

# Calculate option payoffs
K = strike_pct / 100  # Strike price as a fraction of initial price
knock_in_barrier = knock_in_pct / 100  # Knock-in barrier as a fraction of initial price
payoffs = calculate_knockin_worst_of_payoff(S, K, knock_in_barrier)

# Calculate premium and APY
premium_pct, apy = calculate_premium_and_apy(payoffs, notional, r, T)

# Display premium and APY
st.header('Option Premium and APY')
st.write(f'Option Premium: {premium_pct:.2f}% of notional')
st.write(f'Annualized Premium (APY): {apy:.2f}%')

# Plot simulation results
st.header('Simulation Results')
fig, axs = plt.subplots(num_assets, 1, figsize=(12, 5*num_assets), sharex=True)
for i, ticker in enumerate(tickers):
    axs[i].plot(S[:, :, i].T, alpha=0.1, color='blue')
    axs[i].plot(S[:, :, i].mean(axis=0), color='red', linewidth=2)
    axs[i].axhline(y=S0[i] * knock_in_barrier, color='green', linestyle='--', label='Knock-In Barrier')
    axs[i].set_title(f'{ticker} Price Paths')
    axs[i].set_ylabel('Price')
    axs[i].legend()
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
    
    annualized_vol = np.std(returns_sim) * np.sqrt(365)  # Annualized volatility
    
    st.subheader(ticker)
    stringVol = f'Annualized Volatility: {annualized_vol:.2%}'
    if annualized_vol < vol_cap_pct / 100:
        st.write(stringVol)
    else:
        stringVol += f' Volatility capped at: {vol_cap_pct} %'
    st.write(f'Annualized Volatility: {annualized_vol:.2%}')
    st.write(f'95% VaR: {-var_95:.2%}')
    st.write(f'99% VaR: {-var_99:.2%}')
    st.write(f'95% Expected Shortfall: {-es_95:.2%}')

# Payoff distribution
st.header('Option Payoff Distribution')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(payoffs, kde=True, ax=ax)
ax.set_title('Distribution of Option Payoffs')
ax.set_xlabel('Payoff ($)')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Calculate and display knock-in probability
knock_in_prob = np.mean(np.any(np.min(S / S[:, 0, :][:, np.newaxis, :], axis=2) <= knock_in_barrier, axis=1)) * 100
st.header('Knock-In Probability')
st.write(f'Probability of Knock-In: {knock_in_prob:.2f}%')

st.write(f'''
Note: This app prices a European knock-in worst-of put option on the selected cryptocurrencies using a multi-asset Bates model.
The option knocks in if any of the assets touch or go below the knock-in barrier at any time during the option's life.
If knocked in, the payoff at expiration is based on the worst-performing asset.
Initial volatilities are capped at {vol_cap_pct} %.
''')