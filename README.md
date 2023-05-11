![Efficient Frontier](/photos/efficient_frontier.png)

# Introduction: 

## Portfolio_Optimization

The following script will explore the idea of risk vs. return within the context of portfolio optimization. We will be 
constructing a portfolio of assets consisting of the sector ETF’s that represent the 11 major sectors of the S&P.

The question we will be answering is: What is the optimal weighting of each Sector ETF within the portfolio of 11 ETF’s 
that will provide the maximum return, at a given level of risk? How about minimum risk, given level of return? 

This concept is further explained by the Efficient Frontier, whereby we are constructing the optimal portolio where each asset, in this case Sector ETF's, fall onto the curve. This means we have maximized the potential for return, while minimizing the risk/standard deviation of the portfolio. 

There are ten sector ETF's:

Technology - XLK

Health Care - XLV

Financials - XLF

Real Estate - XLRE

Energy - XLE

Materials - XLB

Communication - XLC

Consumer Discretionary - XLY

Consumer Staples - XLP

Financials - XLF

Industrials - XLI

Utilities - XLU

After constructing the portfolio of Sector ETF's, we further explored different assets to see the results of how other securities integrated into a portfolio would produce.

The data is sourced from the ALPACA API to pull closing price historical data (YTD) from each sector ETF.
https://alpaca.markets/stocks

---

## Folders

[The Sector ETF Python Script](/Project_file.ipynb)

[The Other Securities Python Script](/Portfolio_B.ipynb)

[Monte Carlo Simulation Script](/MCForecastTools.py)

---

## Technologies

The script uses Pyton version 3.7. Below is a list of the required installation of the following libraries and dependencies:

> Please see the below links for more information on the following libraries and dependencies!

[Alpaca Trade API](https://pypi.org/project/alpaca-trade-api/0.29/)

[Matplotlib](https://matplotlib.org/)

[Seaborn](https://seaborn.pydata.org/)

[Scipy](https://scipy.org/)

[Dotenv](https://github.com/motdotla/dotenv)

[OS](https://docs.python.org/3/library/os.html)

[Pandas](https://pypi.org/project/pandas/)

[Hvplot](https://pypi.org/project/hvplot/)

[Numpy](https://numpy.org/)

[Pyportfolioopt](https://pypi.org/project/pyportfolioopt/)

---

## Installation Guide

### Step 1: Download the Repo to your local computer.

The easiest way to install the script and corresponding files is to download the remote repo to your local desktop directly from Github.  

> Side Note: The below assumes you will use the command line to download the repo and have SSH keys set up. You can also manually download the file via `Add file` button.

1. To clone, click on the top right button `<> Code` and then copy and paste directly into your command window. 

![Copy Repo](/photos/installation_guide.png)

2. In the command line, use `git clone` and then paste the SSH link. Hit return.

### Part 2: Install the script:

#### Install the required dependencies by running the following command:

```python
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
from dotenv import load_dotenv
import os
import pandas as pd
import hvplot.pandas
import numpy as np
import warnings 
warnings.filterwarnings("ignore")
from MCForecastTools import MCSimulation
%matplotlib inline
```

#### Since we are using an .env file, the script must set the variables for the Alpaca API and Secret Keys 

```python
load_dotenv()

# Grab the Alpaca API Key
alpaca_api_key = os.environ.get('ALPACA_API_KEY')
alpaca_secret_key = os.environ.get('ALPACA_SECRET_KEY')

# Create the Alpaca tradeapi.REST object
alpaca = tradeapi.REST(alpaca_api_key, alpaca_secret_key, api_version = "v2")
```

```python
# Set the tickers you want to use (we are using the 11 Sectors found in the introduction for the purposes of this example)

tickers = ['SPY', 'XLK', 'GLD', 'TLT']

# Set the time frame and format the current date 

timeframe = '1D'

# Format current date as ISO format
start_date = pd.Timestamp("2013-05-01", tz="America/New_York").isoformat()
end_date = pd.Timestamp("2023-05-01", tz="America/New_York").isoformat()
```

### Part 3: Get the portfolio data 

```python
# Using the Alpaca `get_bars()` function to get the currrent closing prices of the portfolio. 
# Be sure to set the `df` property after the function ot format the response object as a DataFrame.

portfolio_data = alpaca.get_bars(
    tickers, 
    timeframe,
    start = start_date,
    end = end_date).df
```

```pyton
# Remove the all columns besides the Closing Price. Iterate through the data so that we extract and create columns to only contain closing prices for each ETF

df = pd.DataFrame()
for ticker in tickers:
    df[ticker] = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == ticker]['close'])
```

#### Part 4: 

We will need to further modify the original `portfolio_data` to run successfully in the Monte Carlo Simulation at the end. To do that now, create a new data frame, `mc_df`.

```python
df1 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[0]])
df2 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[1]])
df3 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[2]])
df4 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[3]])
df5 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[4]])
df6 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[5]])
df7 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[6]])
df8 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[7]])
df9 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[8]])
df10 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[9]])
df11 = pd.DataFrame(portfolio_data[portfolio_data['symbol'] == tickers[10]])
mc_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11], axis = 1, keys = tickers)
mc_df
```

---

## Usage

Running the code will first give us a line plot of the different ETF's price action over the period of 2016-2023. This helps quickly summarize which ETF may produce the highest/lowest returns. 

> The Price Action Line Plot can be seen below. 

![Line Plot of 11 Sector ETF](/photos/line_plot.png)

Even better, checking the daily rate of returns furthers our analysis in finding the most bullish ETFs . 

>  The Daily Rate of Returns can be seen below. 

![Daily Return Plot](/photos/daily_return.png)

### In Part 4 it is necessary to input the various constant values, num_portfolios and risk_free_rate. 

In our example script, we inputed:

```python
# num_portfolios states how many simulations you would like to run.
num_portfolios = 50000

# risk_free_rate is the risk free rate using the 10 year treasury yield
risk_free_rate = 0.0344
```

To display the optimal weights for each ETF 

```python 
display_simulated_ef_with_random (mean_returns, cov_matrix, num_portfolios, risk_free_rate)
```
The result should look something like: 

![Simulated Data Output](/photos/simulated_data.png)

> A plot can be used to see the results visually

![Max_Sharpe](/photos/max_sharpe.png)

---

## Contributors

Novis, Alex
Mbenga, Kayembe
Korman, William
Tadese, Sutan
Justin Valli

SOURCES:

https://www.portfoliovisualizer.com/optimize-portfolio#analysisResults 

https://www.investopedia.com/terms/s/sharperatio.asp 

https://www.youtube.com/watch?v=Usxer0D-WWM (Youtube: How to make an Efficient Frontier Using Python)

https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
---

## License

This project is licensed under the MIT License.


