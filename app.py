import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#---------------------------------#
# Page layout
st.set_page_config(page_title='The Machine Learning Random Forest App', layout='wide')


# Create tabs with larger font size
tab_titles = ['# The New Optimal Model', '# Portfolio Optimization', '# Machine-Learning', '# Results Visualization']
tabs = st.tabs(tab_titles)

#---------------------------------#
# Main panel

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('Upload your CSV dataset'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# sidebar radio button model choice
with st.sidebar:
    Model = st.radio(
        label='Choose a Model', options=('Random Forest', 'Logistic Regression', 'Neural Network'))

# Sidebar - Specify parameter settings
with st.sidebar.header('Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, 50, 50)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (Bootstrapping is a statistical resampling technique that involves random sampling of a dataset with replacement)', options=[True, False])
    #parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the proportion of variance on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

with tabs[0]:
    st.subheader("This endeavor involved a comprehensive refinement and enhancement process applied to our preceding duo of projects: one centered around portfolio optimization and the other delving into the realm of machine learning. Through iterations we honed various aspects of these initiatives and showcased our discoveries via the Streamlit platform.")
    st.write("""# Visualization Performance App

## Shown are the stock, Index or commodity **closing price** and **volume** for:

""")
    col1, col2 = st.columns(2) 
    col1.write("## GLD - Gold")
    col1.write("## USO - United States Oli ETF")
    col1.write("## KO - Coca-Cola")
    col1.write("## GLD - TSLA - Tesla")
    col2.write("## VNQ - Vanguard Real Estate Index ETF")
    col2.write("## AAPL - Apple")
    col2.write("## AGG - iShares Core US Aggregate Bond ETF")
    col2.write("## JNK - SPDR Bloomberg High Yield Bond ETF")

    # Define the ticker symbols
    tickerSymbols = ['GLD', 'VNQ', 'USO', 'KO', 'TSLA', 'AAPL', 'AGG', 'JNK']

    # Get data on these tickers
    tickerData = yf.Tickers(" ".join(tickerSymbols))

    st.write("""## Closing Price""")
    
    # Create a slider to choose ETF for Closing Price
    selected_ticker_close = st.selectbox("Choose for Closing Price", options=tickerSymbols)
    selected_ticker_data_close = tickerData.tickers[selected_ticker_close].history(period='1d', start='2013-05-01', end='2023-05-01')
    st.write(f"### {selected_ticker_close} Closing Price")
    closing_price_chart = st.line_chart(selected_ticker_data_close['Close'])

    st.write("""## Trading Volume""")
    
    # Create a slider to choose ETF for Volume Price
    selected_ticker_volume = st.selectbox("Choose for Volume Price", options=tickerSymbols)
    selected_ticker_data_volume = tickerData.tickers[selected_ticker_volume].history(period='1d', start='2013-05-01', end='2023-05-01')
    st.write(f"### {selected_ticker_volume} Volume Price")
    volume_price_chart = st.line_chart(selected_ticker_data_volume['Volume'])

    # Calculate percentage change in returns for each ETF
    returns = pd.DataFrame({ticker: tickerData.tickers[ticker].history(period='1d', start='2013-05-01', end='2023-05-01')['Close'].pct_change() for ticker in tickerSymbols})

    st.write("""## Percentage Change""")
    
    # Create a slider to choose ETF for Returns
    selected_ticker_returns = st.selectbox("Choose for Returns", options=tickerSymbols)
    st.write(f"### {selected_ticker_returns} Returns")
    selected_ticker_data_returns = returns[selected_ticker_returns]
    st.line_chart(selected_ticker_data_returns)

    

with tabs[1]:
    st.subheader("This project delved into the interplay between risk and return through the lens of portfolio optimization. Our approach involved the assembly of a portfolio comprising sector-specific ETFs, each representing one of the 11 key sectors within the S&P index. Our primary objective was to discern the ideal distribution of weights among these Sector ETFs within the 11-asset portfolio. This distribution aimed to achieve the utmost return potential while adhering to a designated risk threshold. The fundamental underpinning of this concept is embodied by the Efficient Frontier theory. By strategically placing each asset - in this instance, the Sector ETFs - along this curve, we were able to construct a portfolio that is optimized. This optimization entails the maximization of return potential while concurrently minimizing the inherent risk or standard deviation associated with the portfolio.")
    st.write("""# Visualization Performance App

## Shown are the stock **closing price** and **volume** of the ETF's - XLK, XLV, XLF, XLRE, ELE, XLB, XLC, XLY, XLP, XLI, XLU
# """)

    # Define the ticker symbols
    tickerSymbols = ['XLK', 'XLV', 'XLF', 'XLRE', 'ELE', 'XLB', 'XLC', 'XLY', 'XLP', 'XLI', 'XLU']

    # Get data on these tickers
    tickerData = yf.Tickers(" ".join(tickerSymbols))

    st.write("""## ETF's Closing Price""")
    
    # Create a slider to choose ETF for Closing Price
    selected_ticker_close = st.selectbox("Choose ETF for Closing Price", options=tickerSymbols)
    selected_ticker_data_close = tickerData.tickers[selected_ticker_close].history(period='1d', start='2013-05-01', end='2023-05-01')
    st.write(f"### {selected_ticker_close} Closing Price")
    closing_price_chart = st.line_chart(selected_ticker_data_close['Close'])

    st.write("""## ETF's Volume Price""")
    
    # Create a slider to choose ETF for Volume Price
    selected_ticker_volume = st.selectbox("Choose ETF for Volume Price", options=tickerSymbols)
    selected_ticker_data_volume = tickerData.tickers[selected_ticker_volume].history(period='1d', start='2013-05-01', end='2023-05-01')
    st.write(f"### {selected_ticker_volume} Volume Price")
    volume_price_chart = st.line_chart(selected_ticker_data_volume['Volume'])

    # Calculate percentage change in returns for each ETF
    returns = pd.DataFrame({ticker: tickerData.tickers[ticker].history(period='1d', start='2013-05-01', end='2023-05-01')['Close'].pct_change() for ticker in tickerSymbols})

    st.write("""## ETF's Returns""")
    
    # Create a slider to choose ETF for Returns
    selected_ticker_returns = st.selectbox("Choose ETF for Returns", options=tickerSymbols)
    st.write(f"### {selected_ticker_returns} Returns")
    selected_ticker_data_returns = returns[selected_ticker_returns]
    st.line_chart(selected_ticker_data_returns)

    st.image("efficient_frontier_1.png", width=800)
    st.image("scipy_visual.png", width=800)
    st.image("mc_min.png", width=800)

# Model Training tab
with tabs[2]:
  
    if Model == 'Random Forest':
        st.subheader('The **Random Forest Regressor** is used to build a regression model using the **Random Forest** algorithm. Try adjusting the hyperparameters!')
        st.write("## Apple - AAPL")
        #define the ticker symbol
        tickerSymbol ='AAPL'
        #get data on this ticker
        tickerData = yf.Ticker(tickerSymbol)
        #get the historical prices for this ticker
        tickerDf = tickerData.history(period='1d', start='2015-08-01', end='2023-08-01')
        # Open	High	Low	Close	Volume	Dividends	Stock Splits

    
        st.write("""
        ## Closing Price
        """)
        st.line_chart(tickerDf.Close)
        # Displays the dataset
        st.subheader('Dataset')
        

        if uploaded_file is not None:  # Check if a file is uploaded
            df = pd.read_csv(uploaded_file)
            df.reset_index(drop=True, inplace=True)
            df.drop(columns=["Unnamed: 0"], inplace=True)  
            st.markdown('**Sample of dataset**')
            st.write(df)

            # Model building
            X = df.iloc[:,:-1]  # Using all columns except for the last column as X
            Y = df.iloc[:,-1]   # Selecting the last column as Y

            # Data splitting
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)

            st.markdown('**Data splits**')
            st.write('Training set')
            st.info(X_train.shape)
            st.write('Test set')
            st.info(X_test.shape)

            st.markdown('**Variable details**:')
            st.write('X variable')
            st.info(list(X.columns))
            st.write('Y variable')
            st.info(Y.name)

            rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
            random_state=parameter_random_state,
            max_features=parameter_max_features,
            criterion=parameter_criterion,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
            bootstrap=parameter_bootstrap,
            #oob_score=parameter_oob_score,
            n_jobs=parameter_n_jobs)
            rf.fit(X_train, Y_train)

            st.subheader('Model Performance')

            st.markdown('**Training set**')
            Y_pred_train = rf.predict(X_train)
            st.write('Train Accuracy Score:')
            st.info( r2_score(Y_train, Y_pred_train) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_train, Y_pred_train) )

            st.markdown('**Test set**')
            Y_pred_test = rf.predict(X_test)
            st.write('Test Accuracy Score:')
            st.info( r2_score(Y_test, Y_pred_test) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )

            st.subheader('Model Parameters')
            st.write(rf.get_params())
               
        else:
            st.write("Upload a CSV file to start model training.")

    elif Model == 'Logistic Regression':
        st.subheader("Our best model was the Logistic Regression Model. This was in line with our expectations, given the formatting of our dataset and the fact that this is ultimately a classification problem. Please see the image outlining the precision, recall, F1 score, support, and accuracy score of the model.")
        st.image("logistic_regression1.png", width=800)
        st.image("logistic_regression_results.png", width=800)
        #READ IN DATA
        df= pd.read_csv("max_rating1.csv")
        df.head()
        # Select features and target variable
        X = df.drop(columns=["Signal"])
        y = df["Signal"]

        # Split the data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Create a pipeline for preprocessing and modeling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ])

        # Define the hyperparameters grid for grid search
        param_grid = {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l1", "l2"],
        }

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Evaluate the model on training data
        y_train_pred = best_model.predict(X_train)
        train_report = classification_report(y_train, y_train_pred)
        st.subheader("Training Report:")
        st.text(train_report)

        # Evaluate the model on testing data
        y_test_pred = best_model.predict(X_test)
        test_report = classification_report(y_test, y_test_pred)
        st.subheader("Testing Report:")
        st.text(test_report)
    elif Model == 'Neural Network':
        st.subheader("After backtesting and manual optimization, we found the neural network to be less than ideal for predicting the correct BUY or SELL classification. We utilized 2 models. The first model yielded an accuracy score of 0.4701, while the second model gave 0.5203 accuracy score. The difference between the first two models can be found within the hyperparameter tuning. For example, we changed the loss function from Categorical Cross-Entropy to Mean Squared Error, number of hidden nodes from 10 to 20, number of neurons form 2 to 3, and optimizer function from sigmoid to adam.")
        st.image("neural network.png")        


    #---------------------------------#

# Results Visualization tab
with tabs[3]:  
    st.subheader('Results Visualization')
    st.balloons()
    col1, col2 = st.columns(2)   
    col1.header("Original")
    col1.image("efficient_frontier_1.png")
    col1.image("log re.png")
    col1.image("neural network 2.png")
    col2.header("Revised")
    col2.image("efficient_frontier_2.png")
    col2.image("log re 2.png")
    col2.image("neural_network.png")
    col2.image("pairplot.png")
    col2.image("time_series.png")
    col2.image("Heatmap.png")