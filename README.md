# Portfolio_Optimization

The following file will explore the idea of risk vs. return within the context of portfolio optimization. We will be 
constructing a portfolio of assets consisting of the sector ETF’s that represent the 10 major sectors of the S&P.

The question we will be answering is: What is the optimal weighting of each Sector ETF within the portfolio of 10 ETF’s 
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

After we constructed the portfolio of Sector ETF's, we added an additional asset class to experiment with the weightings of the portfolio. 

The data is sourced from the ALPACA API to pull closing price historical data (YTD) from each sector ETF.
https://alpaca.markets/stocks

---

## Technologies

This applicaiton is written in Python and requires the installation of the following libraries:

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os
import pandas as pd
import hvplot.pandas

PYPORTFOLIOOPT

---

## Installation Guide

1. Clone the repository to your local machine.

![1](clone_repo.png)

2. 


2. Install the required dependencies by running the following command:

# Import the required libraries and dependencies

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os
import pandas as pd
import hvplot.pandas

---

## Usage


1. Run the application by executing the following command:

python project_file.ipynb

2. The application will output what is the optimal weighting of each Sector ETF within the portfolio of 10 ETF’s that will provide the least amount of risk, given some level “x” returns

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


