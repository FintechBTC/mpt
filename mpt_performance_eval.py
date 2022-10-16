import pandas as pd
import numpy as np

# constants being used for this module
num_mc_runs = 500
num_mc_years = 5
initial_test_investment = 10000.00
historical_days_percent = 0.75
test_days_percent = 1 - historical_days_percent

####################################################################
# Compute the returns of a portfolio given the portfolio weights and 
# returns of the individual assets
####################################################################
def mpt_pe_compute_portfolio_returns(asset_prices_returns_df, portfolio_volatility):
    # Convert to dataframe (and take the transpose to make the matrix multiplication line up rows vs/ columns)
    portfolio_volatility_df = pd.DataFrame(portfolio_volatility).T
    # extract the weights of the portfolio (drop non weight data)
    portfolio_weights_df = portfolio_volatility_df.drop(['Returns', 'Volatility', 'SharpeRatio'], axis=1, inplace=False)

    return pd.DataFrame(np.dot(asset_prices_returns_df, portfolio_weights_df.T))


####################################################################
# From a sequence of portfolio meta data (returns, volatility, sharpe
# ratio, weights, extract the weigths into a new dataframe
####################################################################
def mpt_pe_get_portfolio_weights_df(portfolio_volatility):
    portfolio_weights = pd.DataFrame(portfolio_volatility).T.drop(['Returns', 'Volatility', 'SharpeRatio'], axis=1, inplace=False).T
    # Change column name to weights
    portfolio_weights.columns =["weights"]
    
    return portfolio_weights


####################################################################
# Arrange Monte Carlo dataframe to have tuples for column names
# as required by the MCForecastTools
####################################################################
def mpt_pe_get_mc_porfolio_df(asset_prices_df):

    # For some reason the MC tools require 2-dimensional column names (at least that's the only way I got it to work)
    # Convert the column names to a tuple (asste name, 'close')
    asset_prices_reformat_col_df = asset_prices_df.copy()
    columns_names_tuple = []
    for column_name in asset_prices_reformat_col_df:
        columns_names_tuple.append((column_name, 'close'))
    asset_prices_reformat_col_df.columns = pd.MultiIndex.from_tuples(columns_names_tuple)
    
    return asset_prices_reformat_col_df


####################################################################
# Arrange Monte Carlo dataframe weights list
# as required by the MCForecastTools
####################################################################
def mpt_pe_get_mc_weights(portfolio_weights):

    # Convert the portfolio weights into a list
    portfolio_weights_dflst = portfolio_weights.to_numpy().tolist()
    # Since the dataframe to list conversion utility creates a 2 dimensional list - could not use this as in MC tools
    # So looped through each weigth in DF and appended to list (to create a single array list)
    portfolio_weights_lst = []
    for list_item in portfolio_weights_dflst:
        portfolio_weights_lst.append(list_item[0])

    return portfolio_weights_lst  
    
####################################################################
# Computer PNL on an initial investment of a portfolio
####################################################################
def mpt_pe_compute_investment_pnl(initial_test_investment, asset_prices_test_beg_df, asset_prices_test_end_df, portfolio_weights_df):
    
    init_investment_per_asset = portfolio_weights_df * initial_test_investment    
    num_shares_per_asset_df = pd.DataFrame(np.round_(np.array(init_investment_per_asset) / np.array(asset_prices_test_beg_df)))    

    return pd.DataFrame(np.dot(asset_prices_test_end_df.T, num_shares_per_asset_df))    
