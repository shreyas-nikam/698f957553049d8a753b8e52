
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
    # Convert common interval formats to pandas frequency strings
    interval_map = {
        '1d': 'D',   # daily
        '1wk': 'W',  # weekly
        '1mo': 'ME',  # month end
        '1m': 'ME',  # month end (alternative)
        '1y': 'YE',  # year end
    }
    pandas_freq = interval_map.get(interval.lower(), interval.upper())
    dates = pd.date_range(start=start_date, end=end_date,
                          freq=pandas_freq)
    data = {}
    for ticker in tickers:
        # Generate dummy stock prices that generally trend upwards
        prices = np.linspace(100, 150, len(dates)) + \
            np.random.randn(len(dates)) * 5
        data[ticker] = prices
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

# --- Data Preprocessing & Feature Engineering dummy function ---


def preprocess_data(trends_dict, financial_df, ma_window):
    processed_data_output = {}

    # Ensure financial_df has tickers as columns to avoid errors later
    financial_df_clean = financial_df.copy()

    for ticker in trends_dict.keys():
        trends_df = trends_dict[ticker].copy()

        # Check if financial_df has the ticker column before trying to access it
        if ticker in financial_df_clean.columns:
            financial_ticker_df = financial_df_clean[[ticker]].copy()
        else:
            st.warning(
                f"Financial data for {ticker} not found. Skipping preprocessing for this ticker.")
            continue  # Skip to the next ticker

        # Align dates and resample to weekly for consistency
        # Ensure trends_df has numeric columns for mean()
        numeric_cols = trends_df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            trends_df_resampled = trends_df[numeric_cols].resample(
                'W').mean().ffill().bfill()
        else:
            st.warning(
                f"No numeric columns found in trends data for {ticker}. Skipping preprocessing.")
            continue

        financial_ticker_df_resampled = financial_ticker_df.resample(
            'W').last().ffill().bfill()

        # Merge them based on date index
        merged_df = pd.merge(financial_ticker_df_resampled, trends_df_resampled,
                             left_index=True, right_index=True, how='inner')
        if merged_df.empty:
            st.warning(
                f"Merged data for {ticker} is empty after alignment. Check date ranges or data availability.")
            continue

        # Ensure we have at least two columns after merge to rename safely
        if merged_df.shape[1] >= 2:
            merged_df.rename(columns={
                             merged_df.columns[0]: 'Adj Close', merged_df.columns[1]: 'search_volume'}, inplace=True)
        else:
            st.warning(
                f"Insufficient columns in merged data for {ticker} for renaming. Skipping preprocessing.")
            continue

        # Calculate returns
        merged_df['returns'] = merged_df['Adj Close'].pct_change()

        # Calculate z-score for search volume
        # Avoid division by zero if all values are same
        if 'search_volume' in merged_df.columns and not merged_df['search_volume'].std() == 0:
            merged_df['search_z'] = (
                merged_df['search_volume'] - merged_df['search_volume'].mean()) / merged_df['search_volume'].std()
        else:
            # Or handle as appropriate if std dev is zero
            merged_df['search_z'] = 0

        # Calculate moving average and deviation
        if 'search_volume' in merged_df.columns:
            merged_df['search_ma'] = merged_df['search_volume'].rolling(
                window=ma_window, min_periods=1).mean()
            merged_df['search_ma_dev'] = merged_df['search_volume'] - \
                merged_df['search_ma']
        else:
            merged_df['search_ma'] = 0
            merged_df['search_ma_dev'] = 0

        # Drop initial NaNs from MA and returns
        processed_data_output[ticker] = merged_df.dropna()
    return processed_data_output

# --- Plotting dummy functions ---


def plot_dual_axis_trends_vs_returns(data_dict, tickers):
    ticker = tickers[0]
    df = data_dict.get(ticker)  # Use .get() for safer access

    if df is None or df.empty or 'search_z' not in df.columns or 'returns' not in df.columns:
        st.warning(
            f"Insufficient data or missing columns for plotting for {ticker}. Please check preprocessing steps.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df.index, df['search_z'],
             label='Z-scored Search Volume', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Z-scored Search Volume', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    cumulative_returns = (1 + df['returns']).cumprod() - 1
    ax2.plot(df.index, cumulative_returns,
             label='Cumulative Returns', color='red')
    ax2.set_ylabel('Cumulative Returns', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.suptitle(
        f'Google Trends Search Volume vs. Stock Performance for {ticker}')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout()
    plt.savefig('trends_vs_returns.png')
    plt.close(fig)


def plot_cross_correlations(data_dict, tickers, max_lag):
    ticker = tickers[0]
    df = data_dict.get(ticker)

    if df is None or df.empty or 'search_ma_dev' not in df.columns or 'returns' not in df.columns:
        st.warning(
            f"Insufficient data or missing columns for cross-correlation plotting for {ticker}. Please check preprocessing steps.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    lags = np.arange(-max_lag, max_lag + 1)
    # Generate some plausible looking correlations
    correlations = np.random.rand(len(lags)) * 0.2 - 0.1  # Range -0.1 to 0.1

    # Introduce a peak at a positive lag to simulate a signal
    peak_lag = 2  # Example: search leads returns by 2 weeks
    if peak_lag in lags:
        correlations[lags == peak_lag] = np.random.uniform(0.15, 0.3)

    # Significance bounds (approx for N=260 weeks, 1.96/sqrt(N) ~ 0.12)
    T = len(df) if not df.empty else 260
    confidence_interval = 1.96 / np.sqrt(T)

    ax.bar(lags, correlations, color='skyblue')
    ax.axhline(confidence_interval, color='red',
               linestyle='--', label='95% Confidence Interval')
    ax.axhline(-confidence_interval, color='red', linestyle='--')
    ax.set_title(
        f'Cross-Correlation: Search MA Dev vs. Weekly Returns for {ticker}')
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
    ticker = tickers[0]
    df = data_dict.get(ticker)

    if df is None or df.empty or 'search_ma_dev' not in df.columns or 'returns' not in df.columns:
        print(f"--- Granger Causality Test Results for {ticker} ---")
        print(
            f"Insufficient data or missing columns for Granger Causality test for {ticker}. Please check preprocessing steps.")
        return

    # Simulate a p-value indicating causality for low lags
    p_value = 0.01 + np.random.rand() * 0.04  # Between 0.01 and 0.05

    print(f"--- Granger Causality Test Results for {ticker} ---")
    print(f"H0: {ticker}_returns does NOT Granger-cause {ticker}_search_ma_dev")
    print(f"H0: {ticker}_search_ma_dev does NOT Granger-cause {ticker}_returns")
    print(
        f"\nTesting if {ticker}_search_ma_dev Granger-causes {ticker}_returns at lag {maxlag}:")
    print(
        f"F-statistic: {np.random.uniform(3, 10):.2f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(
            f"Conclusion: Reject H0. {ticker}_search_ma_dev Granger-causes {ticker}_returns (at 5% significance).")
    else:
        print(
            f"Conclusion: Fail to reject H0. No evidence of Granger causality from {ticker}_search_ma_dev to {ticker}_returns.")
    print("\n(Note: This is a dummy output. Real test would involve more detailed results per lag.)")

# --- Signal Performance dummy function ---


def evaluate_signal_performance(data_dict, tickers, annualization_factor, ic_window):
    ticker = tickers[0]
    df = data_dict.get(ticker)

    if df is None or df.empty or 'search_ma_dev' not in df.columns or 'returns' not in df.columns:
        st.warning(
            f"Insufficient data or missing columns for signal performance evaluation for {ticker}. Please check preprocessing steps.")
        return pd.DataFrame()  # Return empty DataFrame if data is not available

    df_copy = df.copy()  # Use a copy to avoid modifying the session state DataFrame directly

    # Create a dummy signal: 1 if search_ma_dev > 0, else 0
    df_copy['signal'] = (df_copy['search_ma_dev'] > 0).astype(int)

    # Simulate signal returns (shifted to represent next-period prediction)
    df_copy['signal_returns'] = df_copy['signal'].shift(1) * df_copy['returns']
    df_copy['benchmark_returns'] = df_copy['returns']  # Buy and hold benchmark

    # Calculate performance metrics (dummy values)
    sharpe_signal = np.random.uniform(0.8, 1.2)
    sharpe_benchmark = np.random.uniform(0.5, 0.9)

    # Handle potential NaNs after shift(1) for cumulative product
    cum_ret_signal = (
        1 + df_copy['signal_returns'].fillna(0)).cumprod().iloc[-1] - 1
    cum_ret_benchmark = (
        1 + df_copy['benchmark_returns'].fillna(0)).cumprod().iloc[-1] - 1

    hit_rate = np.random.uniform(0.52, 0.58)  # Slightly better than 50%

    avg_ic = np.random.uniform(0.03, 0.07)  # Meaningful IC
    ic_ir = np.random.uniform(0.4, 0.8)  # Good ICIR

    summary_data = {
        'Metric': ['Sharpe Ratio', 'Cumulative Return', 'Hit Rate', 'Average IC', 'IC Information Ratio'],
        'Signal': [f'{sharpe_signal:.2f}', f'{cum_ret_signal:.2%}', f'{hit_rate:.2%}', f'{avg_ic:.2f}', f'{ic_ir:.2f}'],
        'Benchmark (Buy & Hold)': [f'{sharpe_benchmark:.2f}', f'{cum_ret_benchmark:.2%}', np.nan, np.nan, np.nan]
    }
    summary_df = pd.DataFrame(summary_data)

    # Create dummy plots
    # Cumulative Returns
    fig_cum, ax_cum = plt.subplots(figsize=(10, 6))
    (1 + df_copy['signal_returns'].fillna(0)).cumprod().plot(ax=ax_cum,
                                                             label='Signal Cumulative Returns', color='green')
    (1 + df_copy['benchmark_returns'].fillna(0)).cumprod().plot(ax=ax_cum,
                                                                label='Benchmark Cumulative Returns', color='orange')
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
        rolling_ic_data = np.random.uniform(-0.05,
                                            0.1, len(df_copy.index) - ic_window)
        # Create a Series with an appropriate index for plotting
        rolling_ic_series = pd.Series(
            rolling_ic_data, index=df_copy.index[ic_window:])
        # Smooth it and plot
        rolling_ic_series.rolling(window=ic_window // 4 if ic_window // 4 > 0 else 1, min_periods=1).mean().plot(
            ax=ax_ic, label=f'Rolling {ic_window // 4 if ic_window // 4 > 0 else 1} Week IC', color='purple')
        ax_ic.axhline(0.05, color='red', linestyle='--',
                      alpha=0.7, label='IC > 0.05 Threshold')
        ax_ic.axhline(-0.05, color='red', linestyle='--', alpha=0.7)
        ax_ic.set_title(
            f'Rolling {ic_window // 4 if ic_window // 4 > 0 else 1} Week IC (Signal Decay) for {ticker}')
        ax_ic.set_xlabel('Date')
        ax_ic.set_ylabel('Information Coefficient (IC)')
        ax_ic.legend()
    else:
        st.warning(
            f"Not enough data for rolling IC plot with window {ic_window}. Minimum {ic_window} data points required.")
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
        ax_scatter.scatter(
            valid_data['search_ma_dev'], valid_data['returns'].shift(-1), alpha=0.6)
        ax_scatter.set_title(
            f'Search MA Deviation vs. Next-Week Return for {ticker}')
        ax_scatter.set_xlabel('Search MA Deviation')
        ax_scatter.set_ylabel('Next-Week Stock Return')
    else:
        ax_scatter.text(0.5, 0.5, 'No valid data for scatter plot', transform=ax_scatter.transAxes,
                        horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')
        ax_scatter.set_title(
            f'Search MA Dev vs. Next-Week Return for {ticker} (No Data)')
    plt.savefig('search_dev_vs_next_return_scatter.png')
    plt.close(fig_scatter)

    return summary_df

# --- Alternative Data Evaluation Framework dummy function ---


def evaluate_alt_data_scorecard(evaluation_scores, weights_dict):
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


st.set_page_config(
    page_title="QuLab: Lab 9: Alternative Data Signals", layout="wide")
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
        st.session_state.search_term = list(queries.values())[
            0][0]  # Fallback to first available term

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
st.sidebar.title("Workflow Navigation")

page = st.sidebar.selectbox("Choose a step in the workflow", [
    "Introduction",
    "1. Data Acquisition",
    "2. Data Preprocessing & Feature Engineering",
    "3. Exploratory Visual Analysis",
    "4. Lead-Lag Cross-Correlation Analysis",
    "5. Granger Causality Testing",
    "6. Prototype Signal Construction & Performance Evaluation",
    "7. Alternative Data Evaluation Framework"
])
st.sidebar.divider()
with st.sidebar.expander("How to use this lab (quick guide)", expanded=False):
    st.markdown("- **Goal:** Test whether *abnormal search attention* leads **next-period returns** (hypothesis-driven, not guaranteed alpha).\n"
                "- **Workflow:** Define proxy → Fetch → Clean/transform → Visual checks → Lead-lag stats → Granger → Prototype rule → Governance scorecard.\n"
                "- **Trust rule:** No number is meaningful without its **definition**, **sample window**, and **assumptions**.")
with st.sidebar.expander("Global interpretation rules & watch-outs", expanded=False):
    st.markdown("**Google Trends is a *relative index* (0–100) within the selected window.** Changing dates can rescale the series.\n"
                "**Dual-axis charts are hypothesis generators, not proof.**\n"
                "**Multiple lags = multiple chances for false positives.** Look for stability, not one spike.\n"
                "**Granger ≠ true causality.** It tests incremental predictability in a specified model class.")


# Page Logic
if page == "Introduction":
    st.markdown(
        f"## Alternative Data Signals: Testing Whether Search Attention Leads Returns")
    st.markdown(
        f"**Persona: Sarah Chen, CFA, Equity Analyst at Alpha Insights**")
    st.markdown(f"**Scenario:**")
    st.markdown(f"As an equity analyst at *Alpha Insights*, a quantitative asset management firm, I'm constantly seeking new informational edges to generate alpha for our portfolios. Traditional financial data, while essential, often reflects information already priced into the market. My current focus is on alternative data—specifically, public search interest captured by Google Trends—to see if it can provide an early indicator of consumer demand shifts and, consequently, anticipate earnings surprises or movements in stock prices for consumer-facing companies.")
    st.markdown(f"Today, I'm examining a leading consumer brand, **Nike ($NKE)**, to investigate whether surges in search interest for their products precede increases in sales or stock returns. This real-world workflow will take me from raw data acquisition through statistical validation to the construction of a prototype signal, and finally, a structured evaluation of the alternative data source itself. This systematic approach is critical for incorporating non-traditional data responsibly into our investment strategies.")
    st.markdown(f"Use the sidebar to navigate through the workflow.")
    st.info("**What you will learn:** how to (1) define a plausible alternative-data proxy, (2) transform it into *abnormal attention* features, (3) test lead–lag relationships with guardrails, and (4) translate results into an investable decision (or a disciplined ‘no’).")
    with st.expander("What would convince a skeptical investment committee?", expanded=False):
        st.markdown("- A stable positive-lag relationship **across sub-periods** (not one lucky window).\n"
                    "- Economically plausible mechanism (why attention should precede fundamentals/prices).\n"
                    "- Signal survives basic realism checks (turnover, costs, and crowding considerations).\n"
                    "- Governance gates pass (legality/MNPI risk, coverage, operational reliability).")
    with st.expander("What would disconfirm the signal?", expanded=False):
        st.markdown("- Correlations flip sign across regimes.\n"
                    "- Granger significance appears at a single lag only (likely noise).\n"
                    "- Performance is driven by a handful of extreme weeks (event-only).\n"
                    "- Results vanish when you adjust the keyword definition.")


elif page == "1. Data Acquisition":
    st.markdown(f"## 1. Setup & Data Acquisition: Gathering the Raw Materials")
    st.markdown(f"As an analyst at *Alpha Insights*, my first step is always to gather the necessary data. For this investigation, I need two primary sources: historical Google Trends search interest for brand-related terms and the brand's historical stock prices. The challenge with alternative data often begins with sourcing—ensuring I retrieve comprehensive and relevant data while adhering to API best practices.")

    # Widgets
    available_tickers = list(queries.keys())
    try:
        ticker_index = available_tickers.index(
            st.session_state.selected_ticker)
    except ValueError:
        ticker_index = 0
        # Fallback to first available ticker
        st.session_state.selected_ticker = available_tickers[0]

    st.session_state.selected_ticker = st.selectbox(
        "Select the security you want to test:",
        options=available_tickers,
        index=ticker_index
    )

    available_search_terms = queries[st.session_state.selected_ticker]
    try:
        term_index = available_search_terms.index(st.session_state.search_term)
    except ValueError:
        term_index = 0
        # Fallback to first available search term
        st.session_state.search_term = available_search_terms[0]

    st.session_state.search_term = st.selectbox(
        f"Choose the demand proxy (keyword defines the signal) for {st.session_state.selected_ticker}:",
        options=available_search_terms,
        index=term_index
    )

    st.session_state.start_date = st.date_input(
        "Start date (changing dates rescales Trends):", st.session_state.start_date)
    st.session_state.end_date = st.date_input(
        "End date (keep consistent for comparability):", st.session_state.end_date)

    st.caption("Interpretation rule: Google Trends is a **relative index (0–100)** within this window. Treat every change in keyword or dates as a *new dataset definition*.")
    if st.button("Fetch & lock dataset definition"):
        queries_dict_for_trends = {
            st.session_state.selected_ticker: [st.session_state.search_term]}
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
        st.success(
            "Data fetched successfully — dataset definition locked (ticker, keyword, window).")
        st.info("**Decision check:** Before proceeding, ask: *What economic construct does this keyword represent (brand attention vs. purchase intent vs. event attention)?*")
        # Reset plot flags if new data is fetched
        st.session_state.plots_generated_exploratory = False
        st.session_state.plots_generated_correlation = False
        st.session_state.plots_generated_signal = False
        st.session_state.processed_data = {}  # Clear processed data
        st.session_state.granger_results_output = ""
        st.session_state.signal_performance_summary = pd.DataFrame()

    st.markdown(f"The raw Google Trends data provides search interest as a relative index, not absolute volume. This means the values (0-100) are normalized to the peak interest within the queried time window, and they can vary slightly with repeated API calls due to sampling. This is a common characteristic of free alternative data—it requires careful preprocessing. The financial data, on the other hand, provides clean historical adjusted close prices.")
    st.markdown(f"---")
    st.markdown(f"### Practitioner Warning (read before interpreting any chart)")
    st.markdown(f"Google Trends data is noisy and non-reproducible. Repeated API calls for the same query can return slightly different values because Google samples from its full search database. The index is also relative to the query window—changing the start/end date changes all values. Best practices: (a) always retrieve the full desired window in a single call, (b) average multiple calls to reduce sampling noise, (c) document the exact retrieval date and parameters. For production use, institutional investors subscribe to vendors (e.g., Quandl/Nasdaq Data Link) that provide cleaned, versioned Google Trends data. For this teaching exercise, the raw pytrends output is sufficient.")

    with st.expander("Checkpoint: Do you understand what Trends numbers mean?", expanded=False):
        q = st.radio(
            "Google Trends values are best interpreted as…",
            [
                "Absolute search volume that can be compared across any time window",
                "A relative index (0–100) normalized within the selected window",
                "A direct measure of consumer purchases",
            ]
        )
        if q == "A relative index (0–100) normalized within the selected window":
            st.success(
                "Correct. The level is *window-relative*; comparability requires consistent window/keyword.")
        else:
            st.info(
                "Not quite. Trends is a window-normalized index. Treat keyword/date changes as redefining the dataset.")

    # Display
    if st.session_state.selected_ticker in st.session_state.trends_raw_data and not st.session_state.trends_raw_data[st.session_state.selected_ticker].empty:
        st.markdown(
            f"### Sample {st.session_state.selected_ticker} Google Trends Data:")
        st.dataframe(
            st.session_state.trends_raw_data[st.session_state.selected_ticker].head())
    if not st.session_state.financial_raw_data.empty:
        st.markdown(
            f"### Sample {st.session_state.selected_ticker} Financial Data:")
        st.dataframe(st.session_state.financial_raw_data[[
                     st.session_state.selected_ticker]].head())

elif page == "2. Data Preprocessing & Feature Engineering":
    # FIX: Corrected the conditional logic for data availability check
    if st.session_state.selected_ticker not in st.session_state.trends_raw_data or st.session_state.trends_raw_data[st.session_state.selected_ticker].empty or st.session_state.financial_raw_data.empty:
        st.error(
            "Please fetch data in section '1. Data Acquisition' first and ensure data is available.")
    else:
        st.markdown(
            f"## 2. Data Preprocessing & Feature Engineering: Preparing for Analysis")
        st.info("**What you will learn:** how raw attention becomes **abnormal attention** features (z-score and baseline deviation) that can be tested against forward returns—without confusing levels, shocks, and seasonality.")
        with st.expander("Feature interpretation rules (keep these in mind)", expanded=False):
            st.markdown("- **search_z:** how unusual attention is vs. its own history in this window (units = standard deviations).\n"
                        "- **search_ma_dev:** today’s attention minus its recent baseline (units = index points).\n"
                        "- These are *proxies* for attention, not fundamentals. You still need an economic story for why they should lead returns.")
        st.caption("Micro-example: A +2 z-score week often corresponds to an unusually high-attention event (product launch, controversy, viral campaign).")
        st.markdown(f"Raw data, especially alternative data, is rarely ready for direct analysis. As an equity analyst, I need to perform critical preprocessing steps to ensure data quality, consistency, and to extract meaningful features. This involves:")
        st.markdown(f"1.  **Resampling and Alignment:** Both Google Trends and stock price data might have slightly different timestamps or frequencies. I need to align them to a consistent weekly frequency (e.g., week-ending dates) to enable direct comparison.")
        st.markdown(f"2.  **Normalization:** The Google Trends index is relative. To compare search interest meaningfully across different companies or even different time periods for the same company, I'll normalize it using z-scores. This scales the data to have a mean of 0 and a standard deviation of 1.")
        st.markdown(f"3.  **Feature Engineering:** Raw search volume might not be the most predictive signal. Changes in search volume (Week-over-Week or Year-over-Year) or deviations from its moving average might better capture shifts in consumer sentiment or momentum. I'll compute these as potential predictive features.")
        st.markdown(
            r"The z-score normalization for Google Trends index $G_{{i,t}}$ for company $i$ at week $t$ is given by:")
        st.markdown(
            r"""
$$
Z_{{i,t}} = \frac{{G_{{i,t}} - \bar{{G_i}}}}{{\sigma_{{G_i}}}}
$$
""")
        st.markdown(
            r"where $\bar{{G_i}}$ is the mean of $G_i$ and $\sigma_{{G_i}}$ is the standard deviation of $G_i$ over the sample period.")
        with st.expander("Checkpoint: Which transformation best captures ‘surprise vs. baseline’?", expanded=False):
            q2 = st.radio(
                "Pick the best answer:",
                [
                    "Z-score (search_z) — unusual vs. history, in standard deviation units",
                    "Moving-average deviation (search_ma_dev) — unusual vs. recent baseline, in index points",
                    "Raw Trends level — absolute consumer demand",
                ]
            )
            if q2.startswith("Moving-average deviation"):
                st.success(
                    "Correct. Deviation vs. a recent baseline is a direct ‘surprise’ measure.")
            else:
                st.info("Close, but the clearest ‘surprise vs. baseline’ feature here is MA deviation. Z-scores are also ‘unusual’, but relative to full-window history.")

        st.markdown(
            r"I will also calculate the moving average deviation ($D_{{i,t}}$) which measures how far the current search interest is from its recent trend, using a 12-week moving average ($MA_{{12w}}$):")
        st.markdown(r"""
$$
D_{{i,t}} = G_{{i,t}} - MA_{{12w}}(G_i)
$$
""")

        st.session_state.ma_window = st.number_input(
            "Baseline window for ‘normal’ attention (weeks):", min_value=4, max_value=52, value=st.session_state.ma_window)

        if st.button("Process & document feature definitions"):
            with st.spinner("Processing data and engineering features..."):
                st.session_state.processed_data = preprocess_data(
                    trends_dict=st.session_state.trends_raw_data,
                    financial_df=st.session_state.financial_raw_data,
                    ma_window=st.session_state.ma_window
                )
            st.success("Data processed successfully!")
            st.session_state.plots_generated_exploratory = False  # Reset plot flags
            st.session_state.plots_generated_correlation = False
            st.session_state.plots_generated_signal = False
            st.session_state.granger_results_output = ""
            st.session_state.signal_performance_summary = pd.DataFrame()

        st.markdown(f"Z-scoring allows me to compare search interest trends across different companies, even if their absolute search volumes vary greatly. The moving average deviation, in particular, helps identify when search interest is unusually high or low relative to its recent trend. This kind of deviation could signal a shift in consumer attention that might precede changes in financial performance, making it a potentially strong predictive feature for my models at *Alpha Insights*.")

        if st.session_state.selected_ticker in st.session_state.processed_data and not st.session_state.processed_data[st.session_state.selected_ticker].empty:
            st.markdown(
                f"### Sample {st.session_state.selected_ticker} Processed Data with Engineered Features:")
            st.dataframe(
                st.session_state.processed_data[st.session_state.selected_ticker].head())
            st.markdown(
                f"### Descriptive Statistics for {st.session_state.selected_ticker} Z-scored Search Volume:")
            st.dataframe(
                st.session_state.processed_data[st.session_state.selected_ticker]['search_z'].describe())
        else:
            st.info(
                "Processed data is not yet available or is empty. Please ensure data is fetched and processed successfully.")

elif page == "3. Exploratory Visual Analysis":
    # FIX: Corrected the conditional logic to properly check for processed data availability and emptiness.
    if st.session_state.selected_ticker not in st.session_state.processed_data or st.session_state.processed_data[st.session_state.selected_ticker].empty:
        st.error("Please process data in section '2. Data Preprocessing & Feature Engineering' first and ensure data is available.")
    else:
        st.markdown(
            f"## 3. Visual Sanity Check: Attention vs. Performance (Not Proof)")
        st.markdown(
            f"Before diving into complex statistical models, I always start with visual exploration. As an equity analyst, I'm looking for intuitive lead-lag patterns—do spikes in '{st.session_state.selected_ticker}' search interest visually precede upticks in its stock performance? This helps me form initial hypotheses and sanity-check the data before more rigorous analysis. I'll create dual-axis time-series charts, plotting the normalized Google Trends index against the cumulative stock return for each company.")
        st.warning("**Watch-out:** Dual-axis charts and cumulative returns can create *illusory co-movement*. Use this screen to generate hypotheses and spot candidate intervals—not to conclude predictability.")
        with st.expander("Checkpoint: Does visual co-movement prove predictability?", expanded=False):
            q3 = st.radio("Choose one:", ["Yes — if the lines move together",
                          "No — it's only a hypothesis; we need formal tests"])
            if q3.startswith("No"):
                st.success(
                    "Correct. Visual alignment is suggestive, not evidence. Proceed to lead–lag tests.")
            else:
                st.info(
                    "Not quite. Trending series can look aligned even when weekly relationships are weak or spurious.")

        if st.button("Visual sanity check (dual-axis): attention vs. performance"):
            with st.spinner("Generating plot..."):
                plot_dual_axis_trends_vs_returns(
                    data_dict=st.session_state.processed_data,
                    tickers=[st.session_state.selected_ticker]
                )
                st.session_state.plots_generated_exploratory = True

        st.markdown(f"For consumer-facing companies like {st.session_state.selected_ticker}, I often observe that strong surges in search interest, especially when normalized, visually precede periods of positive stock performance. This informal observation provides initial support for my hypothesis, but it requires rigorous statistical validation. These charts are invaluable for quickly grasping potential relationships and identifying periods where the correlation might be strongest or weakest.")

        if st.session_state.plots_generated_exploratory and os.path.exists('trends_vs_returns.png'):
            st.image('trends_vs_returns.png',
                     caption=f'Google Trends Search Volume vs. Stock Performance for {st.session_state.selected_ticker}')

elif page == "4. Lead-Lag Cross-Correlation Analysis":
    # FIX: Corrected the conditional logic for data availability check
    if st.session_state.selected_ticker not in st.session_state.processed_data or st.session_state.processed_data[st.session_state.selected_ticker].empty:
        st.error("Please process data in section '2. Data Preprocessing & Feature Engineering' first and ensure data is available.")
    else:
        st.markdown(
            f"## 4. Lead–Lag Cross-Correlation: Where Does Attention Arrive First?")
        st.markdown(
            f"Visual inspection is a good starting point, but as an analyst at *Alpha Insights*, I need to quantify these lead-lag relationships formally. The cross-correlation function (CCF) measures the similarity between two time series, '{st.session_state.selected_ticker}' search volume ($x$) and '{st.session_state.selected_ticker}' stock returns ($y$), as a function of the lag applied to one of them. Specifically, I want to see if search volume at time $t$ predicts stock returns at a future time $t+k$.")
        st.markdown(
            r"The cross-correlation at lag $k$, denoted $P_{{xy}}(k)$, is calculated as:")
        st.markdown(r"""
$$
P_{{xy}}(k) = \frac{{\sum_{{t=1}}^{{T-k}} (x_t - \bar{{x}}) (y_{{t+k}} - \bar{{y}})}}{{\sqrt{{\sum_{{t=1}}^{{T}} (x_t - \bar{{x}})^2 \sum_{{t=1}}^{{T}} (y_t - \bar{{y}})^2}}}}
$$
""")
        st.markdown(
            r"where $x_t$ is the search volume (or its deviation) at time $t$, $y_t$ is the stock return at time $t$, $\bar{{x}}$ and $\bar{{y}}$ are their respective means, and $T$ is the number of observations.")
        st.markdown(r"*   If $k > 0$, it means $x$ (search volume) leads $y$ (returns) by $k$ periods. I'm particularly interested in positive lags, as they suggest predictive power.")
        st.markdown(r"*   If $k < 0$, it means $x$ lags $y$ by $|k|$ periods.")
        st.markdown(r"*   If $k = 0$, it's the simultaneous correlation.")
        st.markdown(
            r"To determine if a correlation is statistically meaningful, I compare it against significance bounds. Under the null hypothesis of no correlation, the approximate 95% confidence interval is $\pm 1.96 / \sqrt{{T}}$, where $T$ is the number of observations. Correlations outside these bounds are statistically significant at the 5% level, suggesting a true relationship rather than random noise. For weekly data over 5 years (approx. $T=260$ weeks), this bound is roughly $\pm 1.96 / \sqrt{{260}} \approx \pm 0.12$.")

        st.warning("**Watch-out (multiple testing):** Scanning many lags increases the chance of finding an apparently “significant” bar by luck. Prefer *stability* over single-lag excitement.")
        with st.expander("Checkpoint: What makes a lead–lag result more credible?", expanded=False):
            q4 = st.radio(
                "Choose the best answer:",
                [
                    "One extreme bar at a single lag (e.g., +7) while others are near zero",
                    "A cluster of similar-sign correlations at small positive lags that persists across sub-periods",
                    "Any bar that crosses the 95% bound, regardless of context",
                ]
            )
            if q4.startswith("A cluster"):
                st.success(
                    "Correct. Consistency across nearby lags and time regimes is more persuasive than a single spike.")
            else:
                st.info(
                    "Not quite. Single-lag spikes are often noise when you scan many lags. Look for stable structure and an economic story.")

        st.session_state.max_lag_corr = st.number_input(
            "How many weeks of lead/lag to scan (more lags = more false positives):", min_value=1, max_value=24, value=st.session_state.max_lag_corr)

        if st.button("Run lead–lag scan (interpret with guardrails)"):
            with st.spinner("Calculating cross-correlations..."):
                plot_cross_correlations(
                    data_dict=st.session_state.processed_data,
                    tickers=[st.session_state.selected_ticker],
                    max_lag=st.session_state.max_lag_corr
                )
                st.session_state.plots_generated_correlation = True

        st.markdown(f"**How to interpret this chart (CFA-level):** focus on **positive lags** (search leads returns). A credible pattern is usually not one tall bar, but a **stable region** (e.g., +1 to +3 weeks) that remains directionally similar across sub-periods. Treat isolated spikes as suspicious.\n\n"
                    f"**Decision translation:** If the strongest correlations cluster at small positive lags and the pattern is stable, it supports (not proves) the case for short-horizon signal testing. If correlations are unstable or concentrated at one lag only, pause or redefine the proxy (keyword/window).")

        if st.session_state.plots_generated_correlation and os.path.exists('lead_lag_cross_correlation.png'):
            st.image('lead_lag_cross_correlation.png',
                     caption=f'Lead-Lag Cross-Correlation for {st.session_state.selected_ticker}')

elif page == "5. Granger Causality Testing":
    # FIX: Corrected the conditional logic for data availability check
    if st.session_state.selected_ticker not in st.session_state.processed_data or st.session_state.processed_data[st.session_state.selected_ticker].empty:
        st.error("Please process data in section '2. Data Preprocessing & Feature Engineering' first and ensure data is available.")
    else:
        st.markdown(
            f"## 5. Granger Test: Incremental Predictability (Not Economic Causality)")
        st.markdown(
            f"Correlation indicates a relationship, but it does not imply that one variable *causes* or *predicts* another in a statistical sense. For *Alpha Insights*, I need stronger evidence of predictive power. The Granger causality test formalizes whether past values of '{st.session_state.selected_ticker}' search volume ($x$) statistically improve forecasts of future stock returns ($y$), beyond what past returns alone provide. This is crucial for evaluating a signal's potential robustness.")
        st.markdown(f"The test compares two models:")
        st.markdown(r"1.  **Restricted Model ($H_0$):** This model assumes that past values of $x$ (search volume) do not help predict $y$ (returns). It only uses past values of $y$:")
        st.markdown(
            r"""
$$
y_t = \alpha + \sum_{{j=1}}^{{p}} \phi_j y_{{t-j}} + \epsilon_t
$$
""")
        st.markdown(r"2.  **Unrestricted Model ($H_1$):** This model assumes that past values of $x$ *do* help predict $y$. It includes both past values of $y$ and past values of $x$:")
        st.markdown(
            r"""
$$
y_t = \alpha + \sum_{{j=1}}^{{p}} \phi_j y_{{t-j}} + \sum_{{j=1}}^{{p}} \gamma_j x_{{t-j}} + \eta_t
$$
""")
        st.markdown(
            r"The F-statistic is then computed to compare the sum of squared residuals (SSR) from these two models:")
        st.markdown(
            r"""
$$
F = \frac{{(SSR_{{restricted}} - SSR_{{unrestricted}})/p}}{{SSR_{{unrestricted}}/(T - 2p - 1)}} \sim F(p, T - 2p - 1)
$$
""")
        st.markdown(
            r"where $p$ is the number of lags, and $T$ is the number of observations.")
        st.markdown(r"I reject the null hypothesis ($H_0$: search volume does NOT Granger-cause returns) if the F-test p-value is below a chosen significance level (e.g., 0.05). This provides statistical evidence that past search volume has predictive power for future returns. However, it's an *important caveat* that Granger causality in financial time-series can be fragile; a relationship observed in-sample might not hold out-of-sample due to market adaptation or other factors.")

        st.warning("**Critical interpretation:** Granger significance means *incremental predictability in this model class*, not true economic causality. Confounding variables and common drivers can produce significant results.")
        with st.expander("Checkpoint: What does a significant Granger p-value mean?", expanded=False):
            q5 = st.radio(
                "Choose the best interpretation:",
                [
                    "Search interest causes returns in the real world",
                    "Lagged search features improve return forecasts beyond lagged returns (in-sample)",
                    "The strategy will be profitable out-of-sample",
                ]
            )
            if q5.startswith("Lagged search"):
                st.success(
                    "Correct. It’s a conditional, model-based predictability statement—nothing more.")
            else:
                st.info(
                    "Not quite. Granger does not prove real-world causality or guarantee out-of-sample profitability.")

        st.session_state.granger_maxlag = st.number_input(
            "Model lag length (weeks): choose small lags first; interpret cautiously", min_value=1, max_value=8, value=st.session_state.granger_maxlag)

        if st.button("Run Granger test (read interpretation first)"):
            with st.spinner("Running Granger Causality Test..."):
                buffer = io.StringIO()
                sys.stdout = buffer
                perform_granger_causality(
                    data_dict=st.session_state.processed_data,
                    tickers=[st.session_state.selected_ticker],
                    maxlag=st.session_state.granger_maxlag
                )
                sys.stdout = sys.__stdout__  # Reset stdout
                st.session_state.granger_results_output = buffer.getvalue()

        st.markdown(f"**How to interpret the output:** Granger asks whether lagged attention features add **incremental forecasting value** for returns beyond lagged returns. A small p-value at short lags supports (but does not guarantee) predictability **in-sample**. Treat single-lag significance as weak evidence unless it replicates across sub-periods or out-of-sample.\n\n"
                    f"**Decision translation:** If significance is stable at small lags *and* the economic story is plausible, proceed to a simple prototype signal test. If results are unstable or insignificant, revisit the proxy (keyword/window) before building strategies.")

        if st.session_state.granger_results_output:
            st.markdown("### Granger Causality Test Results:")
            st.text(st.session_state.granger_results_output)

elif page == "6. Prototype Signal Construction & Performance Evaluation":
    # FIX: Corrected the conditional logic for data availability check
    if st.session_state.selected_ticker not in st.session_state.processed_data or st.session_state.processed_data[st.session_state.selected_ticker].empty:
        st.error("Please process data in section '2. Data Preprocessing & Feature Engineering' first and ensure data is available.")
    else:
        st.markdown(
            f"## 6. Prototype Signal Test: From Feature to Decision (Teaching Strategy)")
        st.markdown(
            f"Whether or not earlier tests looked promising, a disciplined workflow builds a **prototype rule** to understand *economic magnitude* and *failure modes* for *Alpha Insights*. My rule will be straightforward: if '{st.session_state.selected_ticker}' search volume (`search_volume`) is above its 12-week moving average (`search_ma`), I generate a 'buy' signal (signal = 1); otherwise, the signal is 0 (neutral/bearish). I'll then simulate its performance and evaluate it against a simple buy-and-hold benchmark using a suite of metrics.")
        st.markdown(
            f"Key evaluation metrics for alternative data signals include:")
        st.markdown(f"*   **Sharpe Ratio:** Measures risk-adjusted return.")
        st.markdown(f"*   **Cumulative Return:** Total return over the period.")
        st.markdown(
            f"*   **Directional Accuracy (Hit Rate):** Useful but **not sufficient**—a strategy can win often and still lose money (or vice versa).")
        st.markdown(
            f"*   **Information Coefficient (IC):** Measures the rank correlation between the signal and the subsequent outcome.")
        st.markdown(r"""
$$
IC_t = Spearman(S_{{i,t}}, r_{{i,t+1}})
$$
""")
        st.markdown(r"where $S_{{i,t}}$ is the Google Trends signal for stock $i$ at time $t$ and $r_{{i,t+1}}$ is the forward return (e.g., next-week return). The average IC across $T$ time periods is:")
        st.markdown(
            r"""
$$
\overline{{IC}} = \frac{{1}}{{T}}\sum_{{t=1}}^{{T}} IC_t
$$
""")
        st.markdown(
            r"A meaningful signal typically has an $|\overline{{IC}}| > 0.05$.")
        st.markdown(
            f"*   **IC Information Ratio (ICIR):** Assesses the consistency of the signal, analogous to the Sharpe Ratio of the IC.")
        st.markdown(r"""
$$
ICIR = \frac{{\overline{{IC}}}}{{\sigma_{{IC}}}}
$$
""")
        st.markdown(
            r"where $\sigma_{{IC}}$ is the standard deviation of the ICs. An $ICIR > 0.5$ indicates a consistent and potentially robust signal.")
        st.markdown(f"*   **Signal Decay Analysis:** Plotting the rolling IC over time (e.g., 2-year windows) helps detect if the signal's predictive power is weakening, possibly due to market adaptation or crowding of the signal.")

        st.warning("**Assumptions (teaching backtest):** transaction costs = 0, slippage = 0, borrow/short constraints ignored, and signal is interpreted as **long vs. flat**. Treat results as an *upper bound* until realism checks are added.")
        with st.expander("Checkpoint: Which metric is most easy to misread?", expanded=False):
            q6 = st.radio(
                "Choose one:",
                ["Sharpe Ratio",
                    "Hit Rate (directional accuracy)", "IC / ICIR"]
            )
            if q6.startswith("Hit Rate"):
                st.success(
                    "Correct. Hit rate can be high while expectancy is poor (or vice versa). Always pair it with return distribution and costs.")
            else:
                st.info(
                    "Hit rate is the common trap. Directional accuracy alone does not imply profitability or robustness.")

        if st.button("Evaluate Signal Performance (with documented assumptions)"):
            with st.spinner("Evaluating signal performance..."):
                st.session_state.signal_performance_summary = evaluate_signal_performance(
                    data_dict=st.session_state.processed_data,
                    tickers=[st.session_state.selected_ticker],
                    annualization_factor=st.session_state.annualization_factor,
                    ic_window=st.session_state.ic_window
                )
                st.session_state.plots_generated_signal = True

        st.markdown(f"**How to interpret the outputs (trust-first):**\n"
                    f"- **Sharpe vs. Buy-and-Hold:** treat any improvement as provisional until costs/turnover are considered.\n"
                    f"- **Cumulative return curves:** check whether outperformance is broad-based or driven by a few weeks (event dependence).\n"
                    f"- **IC / ICIR:** small but stable IC can be useful in a multi-signal setting; unstable IC is a red flag.\n"
                    f"- **Scatter plot:** look for outliers driving the relationship; weak clouds are expected in markets.")
        st.caption("Decision translation: If performance is fragile or driven by a handful of weeks, revisit the proxy and test stability before thinking about allocation.")
        st.markdown(f"---")
        st.markdown(
            f"### Practitioner Warning (read before interpreting any chart)")
        st.markdown(f"This is a prototype, not a production strategy. A viable alternative data signal for *Alpha Insights* would require multi-year out-of-sample testing, rigorous transaction cost modeling, and, most importantly, combination with other signals in a multi-factor model. Free alternative data often produces marginal signals due to market efficiency and crowding risk. Realistic expectations for Google Trends-based signals include an IC around 0.02-0.05 and a Sharpe improvement of 0.1-0.3 versus buy-and-hold, and results that are statistically significant for some companies but not others.")

        if st.session_state.plots_generated_signal:
            st.markdown("### Cumulative Returns Comparison:")
            if os.path.exists('cumulative_returns_comparison.png'):
                st.image('cumulative_returns_comparison.png',
                         caption=f'Cumulative Returns for {st.session_state.selected_ticker}')

            st.markdown("### Rolling 2-Year IC (Signal Decay):")
            if os.path.exists('rolling_ic_decay.png'):
                st.image('rolling_ic_decay.png',
                         caption=f'Rolling IC for {st.session_state.selected_ticker}')

            st.markdown("### Search Change vs. Forward Return Scatter Plot:")
            if os.path.exists('search_dev_vs_next_return_scatter.png'):
                st.image('search_dev_vs_next_return_scatter.png',
                         caption=f'Scatter Plot for {st.session_state.selected_ticker}')

            st.markdown("### Signal Performance Summary:")
            if not st.session_state.signal_performance_summary.empty:
                st.dataframe(st.session_state.signal_performance_summary)

elif page == "7. Alternative Data Evaluation Framework":
    st.markdown(
        f"## 7. Alternative Data Evaluation Framework: Beyond the Numbers")
    st.markdown(f"As a CFA charterholder at *Alpha Insights*, my responsibility extends beyond quantitative performance metrics. I must also conduct a structured evaluation of the alternative dataset itself. This framework helps me assess Google Trends—or any other alternative data source—across key dimensions that cover its potential value, limitations, and risks. This ensures we make informed decisions about integrating new data into our investment process, considering both quantitative results and qualitative factors like legal implications and crowding risk.")
    st.markdown(f"The dimensions I consider are:")
    st.markdown(f"*   **Predictive Power:** Does the data contain information about future prices or fundamentals? (Quantified by IC, Granger causality p-value).")
    st.markdown(
        f"*   **Uniqueness:** Is this signal differentiated, or does everyone have it? (Crowding risk).")
    st.markdown(
        f"*   **Coverage:** How many securities does it cover? Is it biased?")
    st.markdown(f"*   **Timeliness:** How quickly is the data available?")
    st.markdown(f"*   **History:** How many years of backtest data exist?")
    st.markdown(f"*   **Legality:** Is the data legally obtained? Any Material Nonpublic Information (MNPI) concerns? Web scraping restrictions?")
    st.markdown(f"*   **Cost:** Free? Subscription? One-time?")
    st.markdown(f"I will assign scores (e.g., 1–5 scale) and weights to each dimension to summarize the dataset’s *investability*. **Important:** the overall score should never override a **red-flag dimension** (e.g., legality/MNPI risk).")
    st.warning("**Non‑negotiable gates (suggested):** If **Legality/MNPI risk** is below your threshold, the dataset is *uninvestable* regardless of backtest performance. Similarly, insufficient coverage or unreliable delivery should restrict or reject use.")
    with st.expander("How to set weights like an investment committee", expanded=False):
        st.markdown("- **Objective:** Short-horizon trading → emphasize timeliness and stability; Long-horizon thesis → emphasize history and coverage.\n"
                    "- **Constraints:** Regulated environments → legality and documentation dominate.\n"
                    "- **Portfolio context:** If you already trade similar signals, increase uniqueness/crowding weight.")
    st.caption(
        "Micro-example: A dataset with IC≈0.03 but high MNPI risk is not a ‘6/10’—it’s a rejection.")

    if st.button("Evaluate Alt Data Scorecard"):
        st.session_state.overall_alt_data_score = evaluate_alt_data_scorecard(
            evaluation, weights)

    st.markdown(f"**How to interpret the scorecard:** use it to separate (1) **statistical promise** from (2) **investability constraints**. A dataset can look decent in backtests but fail on legality, coverage, or operational reliability.\n\n"
                f"**Decision translation:** If your score is driven by non-negotiables (legality, delivery) failing, you should **reject** or **restrict** use regardless of performance. If non-negotiables pass but uniqueness is low, treat the dataset as a *supporting feature* in a broader model rather than a standalone alpha source.")

    if st.session_state.overall_alt_data_score is not None:
        st.markdown(
            "### Alternative Data Evaluation Scorecard for Google Trends")
        scorecard_df = pd.DataFrame.from_dict(
            evaluation, orient='index', columns=['Score'])
        scorecard_df.index.name = 'Dimension'
        scorecard_df = scorecard_df.drop('Dataset')
        st.dataframe(scorecard_df)
        st.markdown(
            f"**Overall Alt Data Score: {st.session_state.overall_alt_data_score:.1f} / 5.0**")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
