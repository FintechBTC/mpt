import pandas as pd
import numpy as np
from pathlib import Path

# constants being used for this module
num_trading_days = 252
risk_free_rate = 0.02
num_random_portfolio_run = 50000
rolling_window_days = 20

####################################################################
# Get asset prices from .csv files
####################################################################
def mpt_qa_get_asset_prices_csv(filename):
    asset_prices_all_symbols_df = pd.read_csv(
        Path(filename), 
        index_col="Date", 
        parse_dates=True, 
        infer_datetime_format=True
    )

    #symbols_sel = ["MARKET","MSFT","AAPL","IBM","JPM","INTC"]
    #asset_prices_df = asset_prices_all_symbols_df[symbols_sel]

    return asset_prices_all_symbols_df

####################################################################
# Compute the frequency components on asset prices
####################################################################
def mpt_qa_compute_freq_stats(asset_prices_df):
    
    # Compute the asset returns
    #asset_prices_returns_df = asset_prices_df.pct_change().dropna()
    asset_prices_returns_df = pd.DataFrame(np.log(asset_prices_df / asset_prices_df.shift(1))).dropna()
    # Compute the annual average asset returns
    asset_prices_ann_returns_mean = asset_prices_returns_df.mean() * num_trading_days
    # Compute the standard deviation of the assets   
    asset_prices_std_df = asset_prices_returns_df.std()
    # Compute the annualized standard deviation of the assets   
    asset_prices_ann_std_df = asset_prices_std_df * np.sqrt(num_trading_days)
    # Compute the rolling_window standard deviation of the assets   
    asset_prices_std_rolling_df = asset_prices_returns_df.rolling(window = rolling_window_days).std()
    # Compute the asset annual covariance matrix
    asset_prices_ann_cov_mtrx_df = asset_prices_returns_df.cov() * num_trading_days
    # Compute the asset correlation matrix
    asset_prices_ann_corr_mtrx_df = asset_prices_returns_df.corr()
    # Compute the cumulative returns
    asset_prices_cum_returns_df = (1 + asset_prices_returns_df).cumprod() - 1
    # Compute the share ratio 
    asset_prices_sharpe_ratio_df = asset_prices_ann_returns_mean / asset_prices_ann_std_df

    return asset_prices_returns_df,        \
           asset_prices_ann_returns_mean,  \
           asset_prices_std_df,            \
           asset_prices_ann_std_df,        \
           asset_prices_std_rolling_df,    \
           asset_prices_ann_cov_mtrx_df,   \
           asset_prices_ann_corr_mtrx_df,  \
           asset_prices_cum_returns_df,    \
           asset_prices_sharpe_ratio_df


####################################################################
# Determine the efficient frontier components
# (1) Portfolio Returns
# (2) Porfolio Volatility (Risk)
# (3) Sharpe ratio 
####################################################################
def mpt_qa_get_efficient_frontier(asset_prices_df, asset_prices_ann_returns_mean, asset_prices_ann_cov_mtrx_df):

    # Initialize an empty list for storing the portfolio returns
    portfolio_returns = []
    # Initialize an empty list for storing the portfolio volatility
    portfolio_volatility = []
    # Initialize an empty list for storing the portfolio weights
    portfolio_weights = []
    # Initialize an empty list for storing the shapre ratio
    portfolio_sharpe_ratio = []

    # Get the number of assets in the asset datafrme
    num_assets = asset_prices_df.shape[1]    

    for portfolio_idx in range(num_random_portfolio_run):
    
        # Normalize weight    
        # Generate random wieghts
        weights = np.random.random(num_assets)
        # Normalize the weights so that the sum of all the wieghts are = 1
        weights = weights / np.sum(weights)    
        # Add teh portfolio weights to the list of portfolio weights
        portfolio_weights.append(weights)
    
        # Compute the portfolio return (weights of each asset * the return of each asset)
        returns = np.dot(weights, asset_prices_ann_returns_mean)    
        # Add the portfolio return to the list of portfolio returns
        portfolio_returns.append(returns)
    
        # Compute the portfolio variance
        variance = asset_prices_ann_cov_mtrx_df.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        # Compute the portfolio starndard deviation
        std_dev = np.sqrt(variance);
        # Annualize the standard deviation will give the volatility
        volatility = std_dev * np.sqrt(num_trading_days)
        # Append the porfolio volatility to the list of portfolio volatilities
        portfolio_volatility.append(volatility)
    
        # Compute the ssharpe ratio
        sharpe_ratio = (returns - risk_free_rate) / volatility;
        # Append the sharpe ratio to the list of protfolio sharpe ratios
        portfolio_sharpe_ratio.append(sharpe_ratio)        

    # Create a list of the asset from the input asset df
    asset_lst = list(asset_prices_df.columns);

    # Create an empty dataframe
    efficient_frontier_df = pd.DataFrame()

    # Add the return and volitilty to each portfolio/row
    efficient_frontier_df['Returns'] = pd.DataFrame(portfolio_returns)
    efficient_frontier_df['Volatility'] = pd.DataFrame(portfolio_volatility)
    efficient_frontier_df['SharpeRatio'] = pd.DataFrame(portfolio_sharpe_ratio)

    # Create a dataframe of number of portfolios by the wieght of easch asset/symbol
    for portfolio_idx, asset in enumerate(asset_prices_df.columns.tolist()):
       efficient_frontier_df[asset+'_W'] = pd.DataFrame([w[portfolio_idx] for w in  portfolio_weights])

    return efficient_frontier_df


####################################################################
# Select portfolio allocations based on the:
# (1) "minimum" risk
# (2) "maximum" risk
# (3) "optimum sharpe ratio"
####################################################################
def mpt_qa_return_portfolio_allocations(efficient_frontier_df):
    portfolio_min_volatility = efficient_frontier_df.iloc[efficient_frontier_df.loc[:,'Volatility'].idxmin()]
    portfolio_max_volatility = efficient_frontier_df.iloc[efficient_frontier_df.loc[:,'Returns'].idxmax()]
    portfolio_opt_sharpe_ratio = efficient_frontier_df.iloc[efficient_frontier_df.loc[:,'SharpeRatio'].idxmax()]
    
    return portfolio_min_volatility, portfolio_max_volatility, portfolio_opt_sharpe_ratio
