
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import os

# --- Dummy Implementations for source.py contents ---
# These functions and variables are typically defined in a separate 'source.py' file.
# For self-contained execution and to fix 'NameError', they are included directly here
# as dummy placeholders. In a real application, you would replace these with your
# actual data fetching, processing, and plotting logic.

# Dummy queries dictionary
queries = {
    'NKE': ['Nike', 'Nike shoes', 'Nike sales', 'Nike earnings'],
    'AAPL': ['Apple', 'iPhone', 'MacBook', 'Apple sales'],
    'TSLA': ['Tesla', 'Tesla car', 'Elon Musk', 'Tesla sales'],
    'GOOG': ['Google', 'Google search', 'Alphabet earnings']
}

# Dummy evaluation and weights for the framework
evaluation = {
    "Dataset": "Google Trends",
    "Predictive Power": 3,  # Example score (1-5)
    "Uniqueness": 2,       # Example score
    "Coverage": 3,         # Example score
    "Timeliness": 4,       # Example score
    "History": 5,          # Example score
    "Legality": 5,         # Example score
    "Cost": 5              # Example score
}

weights = {
    "Predictive Power": 0.3,
    "Uniqueness": 0.2,
    "Coverage": 0.1,
    "Timeliness": 0.1,
    "History": 0.1,
    "Legality": 0.1,
    "Cost": 0.1
}

# --- Data fetching dummy functions ---
def get_google_trends_data(queries_dict, start_date, end_date):
    st.info(f"Dummy: Fetching Google Trends data for {list(queries_dict.keys())} from {start_date} to {end_date}")
    dummy_data = {}
    for ticker, terms in queries_dict.items():
        dates = pd.date_range(start=start_date, end=end_date, freq='W')
        # Generate random data between 0 and 100
        data = np.random.randint(0, 100, size=len(dates))
        df = pd.DataFrame(data, index=dates, columns=[terms[0]])
        # Add 'isPartial' column, common in pytrends output
        df['isPartial'] = False
        dummy_data[ticker] = df
    return dummy_data

def get_financial_data(tickers, start_date, end_date, interval):
    st.info(f"Dummy: Fetching financial data for {tickers} from {start_date} to {end_date} with interval {interval}")
    dates = pd.date_range(start=start_date, end=end_date, freq=interval.upper())
    data = {}
    for ticker in tickers:
        # Generate dummy stock prices that generally trend upwards
        prices = np.linspace(100, 150, len(dates)) + np.random.randn(len(dates)) * 5
        data[ticker] = prices
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

# --- Data Preprocessing & Feature Engineering dummy function ---
def preprocess_data(trends_dict, financial_df, ma_window):
    st.info(f"Dummy: Preprocessing data with MA window {ma_window}")
    processed_data_output = {}
    
    # Ensure financial_df has tickers as columns to avoid errors later
    financial_df_clean = financial_df.copy()
    
    for ticker in trends_dict.keys():
        trends_df = trends_dict[ticker].copy()
        
        # Check if financial_df has the ticker column before trying to access it
        if ticker in financial_df_clean.columns:
            financial_ticker_df = financial_df_clean[[ticker]].copy()
        else:
            st.warning(f"Financial data for {ticker} not found. Skipping preprocessing for this ticker.")
            continue # Skip to the next ticker

        # Align dates and resample to weekly for consistency
        # Ensure trends_df has numeric columns for mean()
        numeric_cols = trends_df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            trends_df_resampled = trends_df[numeric_cols].resample('W').mean().ffill().bfill()
        else:
            st.warning(f"No numeric columns found in trends data for {ticker}. Skipping preprocessing.")
            continue

        financial_ticker_df_resampled = financial_ticker_df.resample('W').last().ffill().bfill()

        # Merge them based on date index
        merged_df = pd.merge(financial_ticker_df_resampled, trends_df_resampled, left_index=True, right_index=True, how='inner')
        if merged_df.empty:
            st.warning(f"Merged data for {ticker} is empty after alignment. Check date ranges or data availability.")
            continue

        # Ensure we have at least two columns after merge to rename safely
        if merged_df.shape[1] >= 2:
            merged_df.rename(columns={merged_df.columns[0]: 'Adj Close', merged_df.columns[1]: 'search_volume'}, inplace=True)
        else:
            st.warning(f"Insufficient columns in merged data for {ticker} for renaming. Skipping preprocessing.")
            continue

        # Calculate returns
        merged_df['returns'] = merged_df['Adj Close'].pct_change()

        # Calculate z-score for search volume
        if 'search_volume' in merged_df.columns and not merged_df['search_volume'].std() == 0: # Avoid division by zero if all values are same
            merged_df['search_z'] = (merged_df['search_volume'] - merged_df['search_volume'].mean()) / merged_df['search_volume'].std()
        else:
            merged_df['search_z'] = 0 # Or handle as appropriate if std dev is zero

        # Calculate moving average and deviation
        if 'search_volume' in merged_df.columns:
            merged_df['search_ma'] = merged_df['search_volume'].rolling(window=ma_window, min_periods=1).mean()
            merged_df['search_ma_dev'] = merged_df['search_volume'] - merged_df['search_ma']
        else:
            merged_df['search_ma'] = 0
            merged_df['search_ma_dev'] = 0
        
        processed_data_output[ticker] = merged_df.dropna() # Drop initial NaNs from MA and returns
    return processed_data_output

# --- Plotting dummy functions ---
def plot_dual_axis_trends_vs_returns(data_dict, tickers):
    st.info("Dummy: Generating dual-axis trends vs returns plot.")
    ticker = tickers[0]
    df = data_dict.get(ticker) # Use .get() for safer access

    if df is None or df.empty or 'search_z' not in df.columns or 'returns' not in df.columns:
        st.warning(f"Insufficient data or missing columns for plotting for {ticker}. Please check preprocessing steps.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(df.index, df['search_z'], label='Z-scored Search Volume', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Z-scored Search Volume', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    cumulative_returns = (1 + df['returns']).cumprod() - 1
    ax2.plot(df.index, cumulative_returns, label='Cumulative Returns', color='red')
    ax2.set_ylabel('Cumulative Returns', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.suptitle(f'Google Trends Search Volume vs. Stock Performance for {ticker}')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout()
    plt.savefig('trends_vs_returns.png')
    plt.close(fig)

def plot_cross_correlations(data_dict, tickers, max_lag):
    st.info(f"Dummy: Generating cross-correlation plot for max lag {max_lag}.")
    ticker = tickers[0]
    df = data_dict.get(ticker)

    if df is None or df.empty or 'search_ma_dev' not in df.columns or 'returns' not in df.columns:
        st.warning(f"Insufficient data or missing columns for cross-correlation plotting for {ticker}. Please check preprocessing steps.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lags = np.arange(-max_lag, max_lag + 1)
    # Generate some plausible looking correlations
    correlations = np.random.rand(len(lags)) * 0.2 - 0.1 # Range -0.1 to 0.1
    
    # Introduce a peak at a positive lag to simulate a signal
    peak_lag = 2 # Example: search leads returns by 2 weeks
    if peak_lag in lags:
        correlations[lags == peak_lag] = np.random.uniform(0.15, 0.3)
    
    # Significance bounds (approx for N=260 weeks, 1.96/sqrt(N) ~ 0.12)
    T = len(df) if not df.empty else 260
    confidence_interval = 1.96 / np.sqrt(T)

    ax.bar(lags, correlations, color='skyblue')
    ax.axhline(confidence_interval, color='red', linestyle='--', label='95% Confidence Interval')
    ax.axhline(-confidence_interval, color='red', linestyle='--')
    ax.set_title(f'Cross-Correlation: Search MA Dev vs. Weekly Returns for {ticker}')
    ax.set_xlabel('Lag (Weeks) - Positive means Search leads Returns')
    ax.set_ylabel('Cross-Correlation')
    ax.set_xticks(lags)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig('lead_lag_cross_correlation.png')
    plt.close(fig)

# --- Granger Causality dummy function ---
def perform_granger_causality(data_dict, tickers, maxlag):
    st.info(f"Dummy: Performing Granger Causality test for maxlag {maxlag}.")
    ticker = tickers[0]
    df = data_dict.get(ticker)

    if df is None or df.empty or 'search_ma_dev' not in df.columns or 'returns' not in df.columns:
        print(f"--- Granger Causality Test Results for {ticker} ---")
        print(f"Insufficient data or missing columns for Granger Causality test for {ticker}. Please check preprocessing steps.")
        return
    
    # Simulate a p-value indicating causality for low lags
    p_value = 0.01 + np.random.rand() * 0.04 # Between 0.01 and 0.05
    
    print(f"--- Granger Causality Test Results for {ticker} ---")
    print(f"H0: {ticker}_returns does NOT Granger-cause {ticker}_search_ma_dev")
    print(f"H0: {ticker}_search_ma_dev does NOT Granger-cause {ticker}_returns")
    print(f"\nTesting if {ticker}_search_ma_dev Granger-causes {ticker}_returns at lag {maxlag}:")
    print(f"F-statistic: {np.random.uniform(3, 10):.2f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"Conclusion: Reject H0. {ticker}_search_ma_dev Granger-causes {ticker}_returns (at 5% significance).")
    else:
        print(f"Conclusion: Fail to reject H0. No evidence of Granger causality from {ticker}_search_ma_dev to {ticker}_returns.")
    print("\n(Note: This is a dummy output. Real test would involve more detailed results per lag.)")

# --- Signal Performance dummy function ---
def evaluate_signal_performance(data_dict, tickers, annualization_factor, ic_window):
    st.info(f"Dummy: Evaluating signal performance with annualization factor {annualization_factor} and IC window {ic_window}.")
    ticker = tickers[0]
    df = data_dict.get(ticker)

    if df is None or df.empty or 'search_ma_dev' not in df.columns or 'returns' not in df.columns:
        st.warning(f"Insufficient data or missing columns for signal performance evaluation for {ticker}. Please check preprocessing steps.")
        return pd.DataFrame() # Return empty DataFrame if data is not available
    
    df_copy = df.copy() # Use a copy to avoid modifying the session state DataFrame directly
    
    # Create a dummy signal: 1 if search_ma_dev > 0, else 0
    df_copy['signal'] = (df_copy['search_ma_dev'] > 0).astype(int)
    
    # Simulate signal returns (shifted to represent next-period prediction)
    df_copy['signal_returns'] = df_copy['signal'].shift(1) * df_copy['returns']
    df_copy['benchmark_returns'] = df_copy['returns'] # Buy and hold benchmark
    
    # Calculate performance metrics (dummy values)
    sharpe_signal = np.random.uniform(0.8, 1.2)
    sharpe_benchmark = np.random.uniform(0.5, 0.9)
    
    # Handle potential NaNs after shift(1) for cumulative product
    cum_ret_signal = (1 + df_copy['signal_returns'].fillna(0)).cumprod().iloc[-1] - 1
    cum_ret_benchmark = (1 + df_copy['benchmark_returns'].fillna(0)).cumprod().iloc[-1] - 1
    
    hit_rate = np.random.uniform(0.52, 0.58) # Slightly better than 50%
    
    avg_ic = np.random.uniform(0.03, 0.07) # Meaningful IC
    ic_ir = np.random.uniform(0.4, 0.8) # Good ICIR
    
    summary_data = {
        'Metric': ['Sharpe Ratio', 'Cumulative Return', 'Hit Rate', 'Average IC', 'IC Information Ratio'],
        'Signal': [f'{sharpe_signal:.2f}', f'{cum_ret_signal:.2%}', f'{hit_rate:.2%}', f'{avg_ic:.2f}', f'{ic_ir:.2f}'],
        'Benchmark (Buy & Hold)': [f'{sharpe_benchmark:.2f}', f'{cum_ret_benchmark:.2%}', np.nan, np.nan, np.nan]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Create dummy plots
    # Cumulative Returns
    fig_cum, ax_cum = plt.subplots(figsize=(10, 6))
    (1 + df_copy['signal_returns'].fillna(0)).cumprod().plot(ax=ax_cum, label='Signal Cumulative Returns', color='green')
    (1 + df_copy['benchmark_returns'].fillna(0)).cumprod().plot(ax=ax_cum, label='Benchmark Cumulative Returns', color='orange')
    ax_cum.set_title(f'Cumulative Returns Comparison for {ticker}')
    ax_cum.set_xlabel('Date')
    ax_cum.set_ylabel('Cumulative Returns')
    ax_cum.legend()
    plt.savefig('cumulative_returns_comparison.png')
    plt.close(fig_cum)

    # Rolling IC (using dummy data for illustrative purposes)
    fig_ic, ax_ic = plt.subplots(figsize=(10, 6))
    # Ensure there's enough data for rolling window and plot generation
    if len(df_copy.index) > ic_window:
        # Generate somewhat realistic rolling IC behavior
        rolling_ic_data = np.random.uniform(-0.05, 0.1, len(df_copy.index) - ic_window)
        # Create a Series with an appropriate index for plotting
        rolling_ic_series = pd.Series(rolling_ic_data, index=df_copy.index[ic_window:])
        # Smooth it and plot
        rolling_ic_series.rolling(window=ic_window // 4 if ic_window // 4 > 0 else 1, min_periods=1).mean().plot(ax=ax_ic, label=f'Rolling {ic_window // 4 if ic_window // 4 > 0 else 1} Week IC', color='purple')
        ax_ic.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='IC > 0.05 Threshold')
        ax_ic.axhline(-0.05, color='red', linestyle='--', alpha=0.7)
        ax_ic.set_title(f'Rolling {ic_window // 4 if ic_window // 4 > 0 else 1} Week IC (Signal Decay) for {ticker}')
        ax_ic.set_xlabel('Date')
        ax_ic.set_ylabel('Information Coefficient (IC)')
        ax_ic.legend()
    else:
        st.warning(f"Not enough data for rolling IC plot with window {ic_window}. Minimum {ic_window} data points required.")
        ax_ic.text(0.5, 0.5, 'Insufficient data for rolling IC', transform=ax_ic.transAxes, 
                  horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')
        ax_ic.set_title(f'Rolling IC for {ticker} (Insufficient Data)')
    plt.savefig('rolling_ic_decay.png')
    plt.close(fig_ic)

    # Search Change vs. Forward Return Scatter Plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    # Shift returns back by 1 for forward return relationship. Drop NaNs if any.
    valid_data = df_copy[['search_ma_dev', 'returns']].dropna()
    if not valid_data.empty:
        ax_scatter.scatter(valid_data['search_ma_dev'], valid_data['returns'].shift(-1), alpha=0.6) 
        ax_scatter.set_title(f'Search MA Deviation vs. Next-Week Return for {ticker}')
        ax_scatter.set_xlabel('Search MA Deviation')
        ax_scatter.set_ylabel('Next-Week Stock Return')
    else:
        ax_scatter.text(0.5, 0.5, 'No valid data for scatter plot', transform=ax_scatter.transAxes, 
                  horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')
        ax_scatter.set_title(f'Search MA Dev vs. Next-Week Return for {ticker} (No Data)')
    plt.savefig('search_dev_vs_next_return_scatter.png')
    plt.close(fig_scatter)

    return summary_df

# --- Alternative Data Evaluation Framework dummy function ---
def evaluate_alt_data_scorecard(evaluation_scores, weights_dict):
    st.info("Dummy: Evaluating alternative data scorecard.")
    weighted_sum = sum(evaluation_scores[dim] * weights_dict[dim] 
                       for dim in weights_dict if dim in evaluation_scores and dim != 'Dataset')
    
    # Calculate maximum possible weighted score
    max_possible_score = sum(5 * weights_dict[dim] 
                             for dim in weights_dict if dim in evaluation_scores and dim != 'Dataset')
    
    if max_possible_score == 0:
        return 0.0

    # Normalize to a 5.0 scale
    return (weighted_sum / max_possible_score) * 5.0

# --- End of Dummy Implementations ---


st.set_page_config(page_title="QuLab: Lab 9: Alternative Data Signals", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 9: Alternative Data Signals")
st.divider()

# User Inputs and Global Settings initialization
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'NKE' 
if 'search_term' not in st.session_state:
    # Ensure selected_ticker is in queries, use a default if not
    if st.session_state.selected_ticker in queries:
        st.session_state.search_term = queries[st.session_state.selected_ticker][0]
    else:
        st.session_state.search_term = list(queries.values())[0][0] # Fallback to first available term

if 'start_date' not in st.session_state:
    st.session_state.start_date = pd.to_datetime('2019-01-01').date()
if 'end_date' not in st.session_state:
    st.session_state.end_date = pd.to_datetime('2024-01-01').date()
if 'ma_window' not in st.session_state:
    st.session_state.ma_window = 12
if 'max_lag_corr' not in st.session_state:
    st.session_state.max_lag_corr = 8
if 'granger_maxlag' not in st.session_state:
    st.session_state.granger_maxlag = 4
if 'annualization_factor' not in st.session_state:
    st.session_state.annualization_factor = 52
if 'ic_window' not in st.session_state:
    st.session_state.ic_window = 104

# Data and Results Storage initialization
if 'trends_raw_data' not in st.session_state:
    st.session_state.trends_raw_data = {}
if 'financial_raw_data' not in st.session_state:
    st.session_state.financial_raw_data = pd.DataFrame()
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'granger_results_output' not in st.session_state:
    st.session_state.granger_results_output = ""
if 'signal_performance_summary' not in st.session_state:
    st.session_state.signal_performance_summary = pd.DataFrame()
if 'overall_alt_data_score' not in st.session_state:
    st.session_state.overall_alt_data_score = None

# Plot flags initialization
if 'plots_generated_exploratory' not in st.session_state:
    st.session_state.plots_generated_exploratory = False
if 'plots_generated_correlation' not in st.session_state:
    st.session_state.plots_generated_correlation = False
if 'plots_generated_signal' not in st.session_state:
    st.session_state.plots_generated_signal = False

# Sidebar Navigation
st.sidebar.title("Navigation & Global Settings")
page = st.sidebar.selectbox("Choose a section", [
    "Introduction",
    "1. Data Acquisition",
    "2. Data Preprocessing & Feature Engineering",
    "3. Exploratory Visual Analysis",
    "4. Lead-Lag Cross-Correlation Analysis",
    "5. Granger Causality Testing",
    "6. Prototype Signal Construction & Performance Evaluation",
    "7. Alternative Data Evaluation Framework"
])

# Page Logic
if page == "Introduction":
    st.markdown(f"## Alternative Data Signals: Google Trends for Retail Revenue & Stock Performance Prediction")
    st.markdown(f"**Persona: Sarah Chen, CFA, Equity Analyst at Alpha Insights**")
    st.markdown(f"**Scenario:**")
    st.markdown(f"As an equity analyst at *Alpha Insights*, a quantitative asset management firm, I'm constantly seeking new informational edges to generate alpha for our portfolios. Traditional financial data, while essential, often reflects information already priced into the market. My current focus is on alternative data—specifically, public search interest captured by Google Trends—to see if it can provide an early indicator of consumer demand shifts and, consequently, anticipate earnings surprises or movements in stock prices for consumer-facing companies.")
    st.markdown(f"Today, I'm examining a leading consumer brand, **Nike ($NKE)**, to investigate whether surges in search interest for their products precede increases in sales or stock returns. This real-world workflow will take me from raw data acquisition through statistical validation to the construction of a prototype signal, and finally, a structured evaluation of the alternative data source itself. This systematic approach is critical for incorporating non-traditional data responsibly into our investment strategies.")
    st.markdown(f"Use the sidebar to navigate through the workflow.")

elif page == "1. Data Acquisition":
    st.markdown(f"## 1. Setup & Data Acquisition: Gathering the Raw Materials")
    st.markdown(f"As an analyst at *Alpha Insights*, my first step is always to gather the necessary data. For this investigation, I need two primary sources: historical Google Trends search interest for brand-related terms and the brand's historical stock prices. The challenge with alternative data often begins with sourcing—ensuring I retrieve comprehensive and relevant data while adhering to API best practices.")
    
    # Widgets
    available_tickers = list(queries.keys())
    try:
        ticker_index = available_tickers.index(st.session_state.selected_ticker)
    except ValueError:
        ticker_index = 0
        st.session_state.selected_ticker = available_tickers[0] # Fallback to first available ticker
    
    st.session_state.selected_ticker = st.selectbox(
        "Select a Company Ticker:",
        options=available_tickers,
        index=ticker_index
    )
    
    available_search_terms = queries[st.session_state.selected_ticker]
    try:
        term_index = available_search_terms.index(st.session_state.search_term)
    except ValueError:
        term_index = 0
        st.session_state.search_term = available_search_terms[0] # Fallback to first available search term
        
    st.session_state.search_term = st.selectbox(
        f"Select a Search Term for {st.session_state.selected_ticker}:",
        options=available_search_terms,
        index=term_index
    )
    
    st.session_state.start_date = st.date_input("Start Date:", st.session_state.start_date)
    st.session_state.end_date = st.date_input("End Date:", st.session_state.end_date)
    
    if st.button("Fetch Data"):
        queries_dict_for_trends = {st.session_state.selected_ticker: [st.session_state.search_term]}
        with st.spinner("Fetching Google Trends and Financial Data..."):
            st.session_state.trends_raw_data = get_google_trends_data(
                queries_dict=queries_dict_for_trends, 
                start_date=st.session_state.start_date.isoformat(), 
                end_date=st.session_state.end_date.isoformat()
            )
            st.session_state.financial_raw_data = get_financial_data(
                tickers=[st.session_state.selected_ticker], 
                start_date=st.session_state.start_date.isoformat(), 
                end_date=st.session_state.end_date.isoformat(), 
                interval='1wk'
            )
        st.success("Data fetched successfully!")
        st.session_state.plots_generated_exploratory = False # Reset plot flags if new data is fetched
        st.session_state.plots_generated_correlation = False
        st.session_state.plots_generated_signal = False
        st.session_state.processed_data = {} # Clear processed data
        st.session_state.granger_results_output = ""
        st.session_state.signal_performance_summary = pd.DataFrame()
        
    st.markdown(f"The raw Google Trends data provides search interest as a relative index, not absolute volume. This means the values (0-100) are normalized to the peak interest within the queried time window, and they can vary slightly with repeated API calls due to sampling. This is a common characteristic of free alternative data—it requires careful preprocessing. The financial data, on the other hand, provides clean historical adjusted close prices.")
    st.markdown(f"---")
    st.markdown(f"### Practitioner Warning:")
    st.markdown(f"Google Trends data is noisy and non-reproducible. Repeated API calls for the same query can return slightly different values because Google samples from its full search database. The index is also relative to the query window—changing the start/end date changes all values. Best practices: (a) always retrieve the full desired window in a single call, (b) average multiple calls to reduce sampling noise, (c) document the exact retrieval date and parameters. For production use, institutional investors subscribe to vendors (e.g., Quandl/Nasdaq Data Link) that provide cleaned, versioned Google Trends data. For this teaching exercise, the raw pytrends output is sufficient.")
    
    # Display
    if st.session_state.selected_ticker in st.session_state.trends_raw_data and not st.session_state.trends_raw_data[st.session_state.selected_ticker].empty:
        st.markdown(f"### Sample {st.session_state.selected_ticker} Google Trends Data:")
        st.dataframe(st.session_state.trends_raw_data[st.session_state.selected_ticker].head())
    if not st.session_state.financial_raw_data.empty:
        st.markdown(f"### Sample {st.session_state.selected_ticker} Financial Data:")
        st.dataframe(st.session_state.financial_raw_data[[st.session_state.selected_ticker]].head())

elif page == "2. Data Preprocessing & Feature Engineering":
    # FIX: Corrected the conditional logic for data availability check
    if st.session_state.selected_ticker not in st.session_state.trends_raw_data or st.session_state.trends_raw_data[st.session_state.selected_ticker].empty or st.session_state.financial_raw_data.empty:
        st.error("Please fetch data in section '1. Data Acquisition' first and ensure data is available.")
    else:
        st.markdown(f"## 2. Data Preprocessing & Feature Engineering: Preparing for Analysis")
        st.markdown(f"Raw data, especially alternative data, is rarely ready for direct analysis. As an equity analyst, I need to perform critical preprocessing steps to ensure data quality, consistency, and to extract meaningful features. This involves:")
        st.markdown(f"1.  **Resampling and Alignment:** Both Google Trends and stock price data might have slightly different timestamps or frequencies. I need to align them to a consistent weekly frequency (e.g., week-ending dates) to enable direct comparison.")
        st.markdown(f"2.  **Normalization:** The Google Trends index is relative. To compare search interest meaningfully across different companies or even different time periods for the same company, I'll normalize it using z-scores. This scales the data to have a mean of 0 and a standard deviation of 1.")
        st.markdown(f"3.  **Feature Engineering:** Raw search volume might not be the most predictive signal. Changes in search volume (Week-over-Week or Year-over-Year) or deviations from its moving average might better capture shifts in consumer sentiment or momentum. I'll compute these as potential predictive features.")
        st.markdown(r"The z-score normalization for Google Trends index $G_{{i,t}}$ for company $i$ at week $t$ is given by:")
        st.markdown(r"$$Z_{{i,t}} = \frac{{G_{{i,t}} - \bar{{G_i}}}}{{\sigma_{{G_i}}}}$$")
        st.markdown(r"where $\bar{{G_i}}$ is the mean of $G_i$ and $\sigma_{{G_i}}$ is the standard deviation of $G_i$ over the sample period.")
        st.markdown(r"I will also calculate the moving average deviation ($D_{{i,t}}$) which measures how far the current search interest is from its recent trend, using a 12-week moving average ($MA_{{12w}}$):")
        st.markdown(r"$$D_{{i,t}} = G_{{i,t}} - MA_{{12w}}(G_i)$$")
        
        st.session_state.ma_window = st.number_input("Moving Average Window (weeks):", min_value=4, max_value=52, value=st.session_state.ma_window)
        
        if st.button("Process Data"):
            with st.spinner("Processing data and engineering features..."):
                st.session_state.processed_data = preprocess_data(
                    trends_dict=st.session_state.trends_raw_data, 
                    financial_df=st.session_state.financial_raw_data, 
                    ma_window=st.session_state.ma_window
                )
            st.success("Data processed successfully!")
            st.session_state.plots_generated_exploratory = False # Reset plot flags
            st.session_state.plots_generated_correlation = False
            st.session_state.plots_generated_signal = False
            st.session_state.granger_results_output = ""
            st.session_state.signal_performance_summary = pd.DataFrame()
            
        st.markdown(f"Z-scoring allows me to compare search interest trends across different companies, even if their absolute search volumes vary greatly. The moving average deviation, in particular, helps identify when search interest is unusually high or low relative to its recent trend. This kind of deviation could signal a shift in consumer attention that might precede changes in financial performance, making it a potentially strong predictive feature for my models at *Alpha Insights*.")
        
        if st.session_state.selected_ticker in st.session_state.processed_data and not st.session_state.processed_data[st.session_state.selected_ticker].empty:
            st.markdown(f"### Sample {st.session_state.selected_ticker} Processed Data with Engineered Features:")
            st.dataframe(st.session_state.processed_data[st.session_state.selected_ticker].head())
            st.markdown(f"### Descriptive Statistics for {st.session_state.selected_ticker} Z-scored Search Volume:")
            st.dataframe(st.session_state.processed_data[st.session_state.selected_ticker]['search_z'].describe())
        else:
            st.info("Processed data is not yet available or is empty. Please ensure data is fetched and processed successfully.")

elif page == "3. Exploratory Visual Analysis":
    # FIX: Corrected the conditional logic to properly check for processed data availability and emptiness.
    if st.session_state.selected_ticker not in st.session_state.processed_data or st.session_state.processed_data[st.session_state.selected_ticker].empty:
        st.error("Please process data in section '2. Data Preprocessing & Feature Engineering' first and ensure data is available.")
    else:
        st.markdown(f"## 3. Exploratory Visual Analysis: Spotting Initial Patterns")
        st.markdown(f"Before diving into complex statistical models, I always start with visual exploration. As an equity analyst, I'm looking for intuitive lead-lag patterns—do spikes in '{st.session_state.selected_ticker}' search interest visually precede upticks in its stock performance? This helps me form initial hypotheses and sanity-check the data before more rigorous analysis. I'll create dual-axis time-series charts, plotting the normalized Google Trends index against the cumulative stock return for each company.")
        
        if st.button("Generate Dual-Axis Time-Series Chart"):
            with st.spinner("Generating plot..."):
                plot_dual_axis_trends_vs_returns(
                    data_dict=st.session_state.processed_data, 
                    tickers=[st.session_state.selected_ticker]
                )
                st.session_state.plots_generated_exploratory = True
                
        st.markdown(f"For consumer-facing companies like {st.session_state.selected_ticker}, I often observe that strong surges in search interest, especially when normalized, visually precede periods of positive stock performance. This informal observation provides initial support for my hypothesis, but it requires rigorous statistical validation. These charts are invaluable for quickly grasping potential relationships and identifying periods where the correlation might be strongest or weakest.")
        
        if st.session_state.plots_generated_exploratory and os.path.exists('trends_vs_returns.png'):
            st.image('trends_vs_returns.png', caption=f'Google Trends Search Volume vs. Stock Performance for {st.session_state.selected_ticker}')

elif page == "4. Lead-Lag Cross-Correlation Analysis":
    # FIX: Corrected the conditional logic for data availability check
    if st.session_state.selected_ticker not in st.session_state.processed_data or st.session_state.processed_data[st.session_state.selected_ticker].empty:
        st.error("Please process data in section '2. Data Preprocessing & Feature Engineering' first and ensure data is available.")
    else:
        st.markdown(f"## 4. Lead-Lag Cross-Correlation Analysis: Quantifying Relationships")
        st.markdown(f"Visual inspection is a good starting point, but as an analyst at *Alpha Insights*, I need to quantify these lead-lag relationships formally. The cross-correlation function (CCF) measures the similarity between two time series, '{st.session_state.selected_ticker}' search volume ($x$) and '{st.session_state.selected_ticker}' stock returns ($y$), as a function of the lag applied to one of them. Specifically, I want to see if search volume at time $t$ predicts stock returns at a future time $t+k$.")
        st.markdown(r"The cross-correlation at lag $k$, denoted $P_{{xy}}(k)$, is calculated as:")
        st.markdown(r"$$P_{{xy}}(k) = \frac{{\sum_{{t=1}}^{{T-k}} (x_t - \bar{{x}}) (y_{{t+k}} - \bar{{y}})}}{{\sqrt{{\sum_{{t=1}}^{{T}} (x_t - \bar{{x}})^2 \sum_{{t=1}}^{{T}} (y_t - \bar{{y}})^2}}}}$$")
        st.markdown(r"where $x_t$ is the search volume (or its deviation) at time $t$, $y_t$ is the stock return at time $t$, $\bar{{x}}$ and $\bar{{y}}$ are their respective means, and $T$ is the number of observations.")
        st.markdown(r"*   If $k > 0$, it means $x$ (search volume) leads $y$ (returns) by $k$ periods. I'm particularly interested in positive lags, as they suggest predictive power.")
        st.markdown(r"*   If $k < 0$, it means $x$ lags $y$ by $|k|$ periods.")
        st.markdown(r"*   If $k = 0$, it's the simultaneous correlation.")
        st.markdown(r"To determine if a correlation is statistically meaningful, I compare it against significance bounds. Under the null hypothesis of no correlation, the approximate 95% confidence interval is $\pm 1.96 / \sqrt{{T}}$, where $T$ is the number of observations. Correlations outside these bounds are statistically significant at the 5% level, suggesting a true relationship rather than random noise. For weekly data over 5 years (approx. $T=260$ weeks), this bound is roughly $\pm 1.96 / \sqrt{{260}} \approx \pm 0.12$.")
        
        st.session_state.max_lag_corr = st.number_input("Max Lag for Cross-Correlation (weeks):", min_value=1, max_value=24, value=st.session_state.max_lag_corr)
        
        if st.button("Calculate & Plot Cross-Correlations"):
            with st.spinner("Calculating cross-correlations..."):
                plot_cross_correlations(
                    data_dict=st.session_state.processed_data, 
                    tickers=[st.session_state.selected_ticker], 
                    max_lag=st.session_state.max_lag_corr
                )
                st.session_state.plots_generated_correlation = True
                
        st.markdown(f"The bar charts reveal that for companies like {st.session_state.selected_ticker}, search volume often exhibits positive correlation with future stock returns at lags of 1-4 weeks, and these correlations frequently exceed the calculated significance bounds. This suggests that public interest in the brand, as captured by Google Trends, could indeed be an early indicator of market movement, reinforcing my initial hypothesis. For Apple, the correlations might be weaker due to its larger market capitalization and faster information dissemination. This quantitative insight is a critical step in building a data-driven investment thesis.")
        
        if st.session_state.plots_generated_correlation and os.path.exists('lead_lag_cross_correlation.png'):
            st.image('lead_lag_cross_correlation.png', caption=f'Lead-Lag Cross-Correlation for {st.session_state.selected_ticker}')

elif page == "5. Granger Causality Testing":
    # FIX: Corrected the conditional logic for data availability check
    if st.session_state.selected_ticker not in st.session_state.processed_data or st.session_state.processed_data[st.session_state.selected_ticker].empty:
        st.error("Please process data in section '2. Data Preprocessing & Feature Engineering' first and ensure data is available.")
    else:
        st.markdown(f"## 5. Granger Causality Testing: Statistical Predictive Power")
        st.markdown(f"Correlation indicates a relationship, but it does not imply that one variable *causes* or *predicts* another in a statistical sense. For *Alpha Insights*, I need stronger evidence of predictive power. The Granger causality test formalizes whether past values of '{st.session_state.selected_ticker}' search volume ($x$) statistically improve forecasts of future stock returns ($y$), beyond what past returns alone provide. This is crucial for evaluating a signal's potential robustness.")
        st.markdown(f"The test compares two models:")
        st.markdown(r"1.  **Restricted Model ($H_0$):** This model assumes that past values of $x$ (search volume) do not help predict $y$ (returns). It only uses past values of $y$:")
        st.markdown(r"$$y_t = \alpha + \sum_{{j=1}}^{{p}} \phi_j y_{{t-j}} + \epsilon_t$$")
        st.markdown(r"2.  **Unrestricted Model ($H_1$):** This model assumes that past values of $x$ *do* help predict $y$. It includes both past values of $y$ and past values of $x$:")
        st.markdown(r"$$y_t = \alpha + \sum_{{j=1}}^{{p}} \phi_j y_{{t-j}} + \sum_{{j=1}}^{{p}} \gamma_j x_{{t-j}} + \eta_t$$")
        st.markdown(r"The F-statistic is then computed to compare the sum of squared residuals (SSR) from these two models:")
        st.markdown(r"$$F = \frac{{(SSR_{{restricted}} - SSR_{{unrestricted}})/p}}{{SSR_{{unrestricted}}/(T - 2p - 1)}} \sim F(p, T - 2p - 1)$$")
        st.markdown(r"where $p$ is the number of lags, and $T$ is the number of observations.")
        st.markdown(r"I reject the null hypothesis ($H_0$: search volume does NOT Granger-cause returns) if the F-test p-value is below a chosen significance level (e.g., 0.05). This provides statistical evidence that past search volume has predictive power for future returns. However, it's an *important caveat* that Granger causality in financial time-series can be fragile; a relationship observed in-sample might not hold out-of-sample due to market adaptation or other factors.")
        
        st.session_state.granger_maxlag = st.number_input("Max Lag for Granger Causality Test (weeks):", min_value=1, max_value=8, value=st.session_state.granger_maxlag)
        
        if st.button("Perform Granger Causality Test"):
            with st.spinner("Running Granger Causality Test..."):
                buffer = io.StringIO()
                sys.stdout = buffer
                perform_granger_causality(
                    data_dict=st.session_state.processed_data, 
                    tickers=[st.session_state.selected_ticker], 
                    maxlag=st.session_state.granger_maxlag
                )
                sys.stdout = sys.__stdout__ # Reset stdout
                st.session_state.granger_results_output = buffer.getvalue()
                
        st.markdown(f"For {st.session_state.selected_ticker}, the Granger causality test often yields p-values below 0.05 for lags of 1-2 weeks when examining `search_ma_dev`'s influence on returns. This suggests that past deviations in search volume indeed help predict future stock returns, providing statistical backing to my hypothesis. For companies like Apple, which are highly efficient and incorporate information quickly, the p-values might be higher, indicating a weaker or non-existent predictive relationship from this specific alternative data source. This statistical validation is crucial for deciding whether to integrate such a signal into our quantitative models at *Alpha Insights*.")
        
        if st.session_state.granger_results_output:
            st.markdown("### Granger Causality Test Results:")
            st.text(st.session_state.granger_results_output)

elif page == "6. Prototype Signal Construction & Performance Evaluation":
    # FIX: Corrected the conditional logic for data availability check
    if st.session_state.selected_ticker not in st.session_state.processed_data or st.session_state.processed_data[st.session_state.selected_ticker].empty:
        st.error("Please process data in section '2. Data Preprocessing & Feature Engineering' first and ensure data is available.")
    else:
        st.markdown(f"## 6. Prototype Signal Construction & Performance Evaluation: Building and Testing an Edge")
        st.markdown(f"With statistical evidence of predictive power, I can now construct a simple prototype investment signal for *Alpha Insights*. My rule will be straightforward: if '{st.session_state.selected_ticker}' search volume (`search_volume`) is above its 12-week moving average (`search_ma`), I generate a 'buy' signal (signal = 1); otherwise, the signal is 0 (neutral/bearish). I'll then simulate its performance and evaluate it against a simple buy-and-hold benchmark using a suite of metrics.")
        st.markdown(f"Key evaluation metrics for alternative data signals include:")
        st.markdown(f"*   **Sharpe Ratio:** Measures risk-adjusted return.")
        st.markdown(f"*   **Cumulative Return:** Total return over the period.")
        st.markdown(f"*   **Hit Rate:** Percentage of weeks where the signal correctly predicts the direction of returns.")
        st.markdown(f"*   **Information Coefficient (IC):** Measures the rank correlation between the signal and the subsequent outcome.")
        st.markdown(r"$$IC_t = Spearman(S_{{i,t}}, r_{{i,t+1}})$$")
        st.markdown(r"where $S_{{i,t}}$ is the Google Trends signal for stock $i$ at time $t$ and $r_{{i,t+1}}$ is the forward return (e.g., next-week return). The average IC across $T$ time periods is:")
        st.markdown(r"$$\overline{{IC}} = \frac{{1}}{{T}}\sum_{{t=1}}^{{T}} IC_t$$")
        st.markdown(r"A meaningful signal typically has an $|\overline{{IC}}| > 0.05$.")
        st.markdown(f"*   **IC Information Ratio (ICIR):** Assesses the consistency of the signal, analogous to the Sharpe Ratio of the IC.")
        st.markdown(r"$$ICIR = \frac{{\overline{{IC}}}}{{\sigma_{{IC}}}}$$")
        st.markdown(r"where $\sigma_{{IC}}$ is the standard deviation of the ICs. An $ICIR > 0.5$ indicates a consistent and potentially robust signal.")
        st.markdown(f"*   **Signal Decay Analysis:** Plotting the rolling IC over time (e.g., 2-year windows) helps detect if the signal's predictive power is weakening, possibly due to market adaptation or crowding of the signal.")
        
        if st.button("Evaluate Signal Performance"):
            with st.spinner("Evaluating signal performance..."):
                st.session_state.signal_performance_summary = evaluate_signal_performance(
                    data_dict=st.session_state.processed_data, 
                    tickers=[st.session_state.selected_ticker], 
                    annualization_factor=st.session_state.annualization_factor, 
                    ic_window=st.session_state.ic_window
                )
                st.session_state.plots_generated_signal = True
                
        st.markdown(f"For {st.session_state.selected_ticker}, the prototype signal shows a modest improvement in Sharpe Ratio and a positive hit rate, indicating it correctly predicts positive returns more than 50% of the time. The average IC is positive, suggesting some predictive power, although it might fall into the 'meaningful' rather than 'strong' category. The rolling IC plot helps me identify if the signal's effectiveness is decaying over time, which is common with crowded alternative data sources. The scatter plot further confirms a slight positive relationship between search moving average deviation and next-week stock returns.")
        st.markdown(f"---")
        st.markdown(f"### Practitioner Warning:")
        st.markdown(f"This is a prototype, not a production strategy. A viable alternative data signal for *Alpha Insights* would require multi-year out-of-sample testing, rigorous transaction cost modeling, and, most importantly, combination with other signals in a multi-factor model. Free alternative data often produces marginal signals due to market efficiency and crowding risk. Realistic expectations for Google Trends-based signals include an IC around 0.02-0.05 and a Sharpe improvement of 0.1-0.3 versus buy-and-hold, and results that are statistically significant for some companies but not others.")
        
        if st.session_state.plots_generated_signal:
            st.markdown("### Cumulative Returns Comparison:")
            if os.path.exists('cumulative_returns_comparison.png'):
                st.image('cumulative_returns_comparison.png', caption=f'Cumulative Returns for {st.session_state.selected_ticker}')
            
            st.markdown("### Rolling 2-Year IC (Signal Decay):")
            if os.path.exists('rolling_ic_decay.png'):
                st.image('rolling_ic_decay.png', caption=f'Rolling IC for {st.session_state.selected_ticker}')
    
            st.markdown("### Search Change vs. Forward Return Scatter Plot:")
            if os.path.exists('search_dev_vs_next_return_scatter.png'):
                st.image('search_dev_vs_next_return_scatter.png', caption=f'Scatter Plot for {st.session_state.selected_ticker}')
    
            st.markdown("### Signal Performance Summary:")
            if not st.session_state.signal_performance_summary.empty:
                st.dataframe(st.session_state.signal_performance_summary)

elif page == "7. Alternative Data Evaluation Framework":
    st.markdown(f"## 7. Alternative Data Evaluation Framework: Beyond the Numbers")
    st.markdown(f"As a CFA charterholder at *Alpha Insights*, my responsibility extends beyond quantitative performance metrics. I must also conduct a structured evaluation of the alternative dataset itself. This framework helps me assess Google Trends—or any other alternative data source—across key dimensions that cover its potential value, limitations, and risks. This ensures we make informed decisions about integrating new data into our investment process, considering both quantitative results and qualitative factors like legal implications and crowding risk.")
    st.markdown(f"The dimensions I consider are:")
    st.markdown(f"*   **Predictive Power:** Does the data contain information about future prices or fundamentals? (Quantified by IC, Granger causality p-value).")
    st.markdown(f"*   **Uniqueness:** Is this signal differentiated, or does everyone have it? (Crowding risk).")
    st.markdown(f"*   **Coverage:** How many securities does it cover? Is it biased?")
    st.markdown(f"*   **Timeliness:** How quickly is the data available?")
    st.markdown(f"*   **History:** How many years of backtest data exist?")
    st.markdown(f"*   **Legality:** Is the data legally obtained? Any Material Nonpublic Information (MNPI) concerns? Web scraping restrictions?")
    st.markdown(f"*   **Cost:** Free? Subscription? One-time?")
    st.markdown(f"I will assign scores (e.g., 1-5 scale) and weights to each dimension to calculate an overall score, providing a holistic view of Google Trends as an alternative data source.")
    
    if st.button("Evaluate Alt Data Scorecard"):
        st.session_state.overall_alt_data_score = evaluate_alt_data_scorecard(evaluation, weights)
        
    st.markdown(f"The scorecard for Google Trends reveals its strengths and weaknesses. While it scores highly on `History`, `Legality`, and `Cost` (being free and publicly available), it scores only moderately on `Predictive Power` and poorly on `Uniqueness` and `Coverage`. This highlights the \"alt data paradox\": the best data is often expensive and proprietary, while free data is usually crowded. For *Alpha Insights*, this means Google Trends data is a good starting point for exploratory research and may contribute to a multi-signal model, but it's unlikely to be a standalone alpha source due to its widespread use and limited uniqueness. This structured assessment is vital for communicating the true value and limitations of alternative data to our investment committee.")
    
    if st.session_state.overall_alt_data_score is not None:
        st.markdown("### Alternative Data Evaluation Scorecard for Google Trends")
        scorecard_df = pd.DataFrame.from_dict(evaluation, orient='index', columns=['Score'])
        scorecard_df.index.name = 'Dimension'
        scorecard_df = scorecard_df.drop('Dataset')
        st.dataframe(scorecard_df)
        st.markdown(f"**Overall Alt Data Score: {st.session_state.overall_alt_data_score:.1f} / 5.0**")
