
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

def mpt_precanned_asset_selection(asset_tickers, market_ticker, Start_date, End_date):
    
    ##function accepts: asset_tickers,  as a list of tickers in str format, a market benchmark ticker as a str, an end date as a str and a start date 
    #as a string. 
    ##to use: import mpt_pre_canned_asset_selection from data-collection.py and call function and pass it the aforementioned parameters
    


    alpaca_api_key = "AKEF6S32QWN9RJIQVYDP"


    secret_key = "TG3BiIvp1u1fcaEpcXdXNaNw4u2nPqNfdVH7GXBz"

    alpaca = tradeapi.REST(alpaca_api_key, secret_key, api_version='v2')

  
    symbols = asset_tickers
    market_symbol = market_ticker
    

    timeframe = '1Day'
    start_date = pd.Timestamp(Start_date, tz='America/New_York').isoformat() 
    end_date = pd.Timestamp(End_date,tz='America/New_York').isoformat()

    asset_df = alpaca.get_bars(symbols, timeframe, start_date, end_date).df
    
    market_df = alpaca.get_bars(market_symbol, timeframe, start_date, end_date).df
    asset_df = asset_df.set_index(['symbol'])
    asset_df = asset_df.drop(['vwap', 'high', 'open', 'low', 'trade_count'], axis=1)
    asset_df = asset_df.dropna()
    market_df = market_df.set_index(['symbol'])
    market_df = market_df.drop(['vwap', 'high', 'open', 'low', 'trade_count'], axis=1)
    market_df =  market_df.dropna()

    return asset_df, asset_tickers, market_ticker, market_df
