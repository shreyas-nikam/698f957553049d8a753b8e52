import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import os

# Set global display options for pandas for consistent output if run interactively
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- Configuration Constants ---
# These can be externalized to a config file or passed as function parameters
DEFAULT_QUERIES = {
    'NKE': ['Nike shoes', 'Nike'],
    'WMT': ['Walmart', 'Walmart deals'],
    'AAPL': ['iPhone', 'Apple store'],
    'TSLA': ['Tesla', 'Tesla Model 3'],
    'SBUX': ['Starbucks', 'Starbucks menu']
}
DEFAULT_TICKERS = list(DEFAULT_QUERIES.keys())
DEFAULT_START_DATE = '2019-01-01'
DEFAULT_END_DATE = '2024-01-01'
DEFAULT_INTERVAL = '1wk'
DEFAULT_MA_WINDOW = 12  # For moving average in preprocessing and signal
DEFAULT_MAX_LAG_CORR = 8  # For cross-correlation plotting
DEFAULT_MAX_LAG_GRANGER = 4  # For Granger causality test
DEFAULT_ANNUALIZATION_FACTOR = 52  # Weekly data for Sharpe Ratio (52 weeks in a year)
DEFAULT_IC_WINDOW = 104  # For rolling Information Coefficient (approx 2 years of weekly data)

# --- Alternative Data Evaluation Scorecard Definition ---
ALT_DATA_EVALUATION = {
    'Dataset': 'Google Trends Search Volume',
    'Predictive Power': 3,  # Moderate IC, some Granger causality
    'Uniqueness': 1,      # Low, widely available
    'Coverage': 2,        # Narrow (consumer-facing only)
    'Timeliness': 4,      # Weekly (some daily)
    'History': 5,         # 2004-present (~20 yrs)
    'Legality': 5,        # Legal (public API), no MNPI concerns
    'Cost': 5             # Free (raw data)
}

ALT_DATA_WEIGHTS = {
    'Predictive Power': 0.30,
    'Uniqueness': 0.20,
    'Coverage': 0.15,
    'Timeliness': 0.10,
    'History': 0.10,
    'Legality': 0.10,
    'Cost': 0.05
}

# --- Core Functions ---

def initialize_pytrends():
    """Initializes and returns a pytrends client."""
    return TrendReq(hl='en-US', tz=360)

def get_google_trends_data(pytrends_client: TrendReq, queries_dict: dict,
                           start_date: str = DEFAULT_START_DATE,
                           end_date: str = DEFAULT_END_DATE) -> dict[str, pd.DataFrame]:
    """
    Retrieves Google Trends 'interest over time' data for multiple companies and keywords.
    Includes rate limiting to avoid API blocks.

    Args:
        pytrends_client: An initialized pytrends.request.TrendReq object.
        queries_dict (dict): A dictionary where keys are tickers and values are lists of keywords.
        start_date (str): Start date for trends data (YYYY-MM-DD).
        end_date (str): End date for trends data (YYYY-MM-DD).

    Returns:
        dict: A dictionary where keys are tickers and values are pandas DataFrames
              containing 'search_volume'. Returns an empty dict if no data.
    """
    trends_data = {}
    for ticker, kw_list in queries_dict.items():
        print(f"Retrieving Google Trends data for {ticker}...")
        try:
            # Pytrends API is limited to 5 keywords per call. For simplicity, we take the first keyword.
            pytrends_client.build_payload([kw_list[0]], cat=0, timeframe=f'{start_date} {end_date}', geo='US', gprop='')
            interest = pytrends_client.interest_over_time()
            if not interest.empty:
                # Remove 'isPartial' column if it exists and rename the keyword column to 'search_volume'
                interest = interest.drop(columns=['isPartial'], errors='ignore')
                interest.columns = ['search_volume']
                trends_data[ticker] = interest
            else:
                print(f"No Google Trends data found for {ticker} with keyword '{kw_list[0]}'.")
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
        time.sleep(2) # Rate limiting to avoid API blocks
    return trends_data

def get_financial_data(tickers: list[str], start_date: str = DEFAULT_START_DATE,
                       end_date: str = DEFAULT_END_DATE, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """
    Retrieves historical adjusted close prices for given tickers from Yahoo Finance.

    Args:
        tickers (list): A list of stock tickers.
        start_date (str): Start date for financial data (YYYY-MM-DD).
        end_date (str): End date for financial data (YYYY-MM-DD).
        interval (str): Data interval (e.g., '1d', '1wk', '1mo').

    Returns:
        pd.DataFrame: A DataFrame with 'Close' prices for each ticker, indexed by date.
                      Returns an empty DataFrame if no data is found or an error occurs.
    """
    print(f"\nRetrieving financial data for {tickers}...")
    financial_data = pd.DataFrame() # Initialize as empty DataFrame
    try:
        prices = yf.download(tickers, start=start_date, end=end_date, interval=interval)['Close']

        # If only one ticker, prices will be a Series, convert to DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        financial_data = prices
        print("Financial data retrieval complete.")
    except Exception as e:
        print(f"Error retrieving financial data: {e}")
    return financial_data

def preprocess_data(trends_dict: dict[str, pd.DataFrame], financial_df: pd.DataFrame,
                    ma_window: int = DEFAULT_MA_WINDOW) -> dict[str, pd.DataFrame]:
    """
    Aligns, normalizes, and generates features for Google Trends and financial data.

    Args:
        trends_dict (dict): Dictionary of raw Google Trends DataFrames.
        financial_df (pd.DataFrame): DataFrame of raw financial close prices.
        ma_window (int): Window size for moving average calculation.

    Returns:
        dict: A dictionary where keys are tickers and values are processed pandas DataFrames
              with 'search_volume', 'return', 'search_z', 'search_wow', 'search_ma',
              'search_ma_dev', and 'cumulative_return' columns.
    """
    processed_data = {}

    if not isinstance(financial_df, pd.DataFrame) or financial_df.empty:
        print("Warning: Financial data is not a valid pandas DataFrame or is empty. Skipping preprocessing.")
        return {}

    for ticker, trend_df in trends_dict.items():
        if ticker not in financial_df.columns:
            print(f"Skipping {ticker}: No financial data available.")
            continue

        # 1. Resample and Align
        trend_df.index = pd.to_datetime(trend_df.index)
        trend_weekly = trend_df.resample('W').last().fillna(method='ffill')

        financial_series = financial_df[ticker].resample('W').last().fillna(method='ffill')
        returns = financial_series.pct_change().dropna()

        # Merge on weekly dates
        merged_df = pd.merge(trend_weekly, returns.rename('return'), left_index=True, right_index=True, how='inner')

        # Ensure 'search_volume' is numeric
        merged_df['search_volume'] = pd.to_numeric(merged_df['search_volume'], errors='coerce')
        merged_df = merged_df.dropna(subset=['search_volume', 'return'])

        if merged_df.empty:
            print(f"Skipping {ticker}: Merged DataFrame is empty after alignment.")
            continue

        # 2. Normalize search volume to z-scores
        std_dev = merged_df['search_volume'].std()
        merged_df['search_z'] = (merged_df['search_volume'] - merged_df['search_volume'].mean()) / std_dev if std_dev != 0 else 0.0

        # 3. Feature Engineering
        # Week-over-week change
        merged_df['search_wow'] = merged_df['search_volume'].diff()

        # Moving average deviation (12-week MA)
        merged_df['search_ma'] = merged_df['search_volume'].rolling(window=ma_window).mean()
        merged_df['search_ma_dev'] = merged_df['search_volume'] - merged_df['search_ma']

        # Calculate cumulative return for visualization
        merged_df['cumulative_return'] = (1 + merged_df['return']).cumprod() - 1

        processed_data[ticker] = merged_df.dropna() # Drop rows with NaNs introduced by diff() or rolling()

    return processed_data

def lead_lag_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = DEFAULT_MAX_LAG_CORR) -> tuple[list[int], list[float]]:
    """
    Compute cross-correlation between x and y at lags -max_lag to +max_lag.
    Positive lags mean x leads y.

    Args:
        x (np.ndarray): The leading variable (e.g., search volume deviation).
        y (np.ndarray): The lagging variable (e.g., stock returns).
        max_lag (int): Maximum lag to compute correlation for.

    Returns:
        tuple: A tuple containing (list of lags, list of correlations).
    """
    # Normalize inputs to have mean 0 and std dev 1
    x_norm = (x - np.mean(x)) / np.std(x)
    y_norm = (y - np.mean(y)) / np.std(y)

    lags = list(range(-max_lag, max_lag + 1))
    correlations = []

    for lag in lags:
        if lag >= 0:
            # x leads y (x is current, y is future)
            corr = np.corrcoef(x_norm[:-lag] if lag > 0 else x_norm,
                               y_norm[lag:])[0, 1]
        else:
            # x lags y (x is future, y is current) - this means y leads x
            corr = np.corrcoef(x_norm[-lag:],
                               y_norm[:lag])[0, 1]
        correlations.append(corr)
    return lags, correlations

def plot_cross_correlations(data_dict: dict[str, pd.DataFrame], tickers: list[str],
                            max_lag: int = DEFAULT_MAX_LAG_CORR,
                            output_path: str = 'lead_lag_cross_correlation.png'):
    """
    Generates lead-lag cross-correlation bar charts for specified tickers.

    Args:
        data_dict (dict): Dictionary of processed data for each ticker.
        tickers (list): List of tickers to plot.
        max_lag (int): Maximum lag for correlation calculation.
        output_path (str): File path to save the plot.
    """
    valid_tickers = [t for t in tickers if t in data_dict and not data_dict[t].empty]
    if not valid_tickers:
        print("No valid tickers with processed data found for cross-correlation plotting.")
        return

    num_plots = len(valid_tickers)
    num_rows = int(np.ceil(num_plots / 2)) if num_plots > 2 else num_plots
    num_cols = 2 if num_plots > 1 else 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 5 * num_rows))
    axes = axes.flatten() if num_plots > 1 else [axes]

    fig.suptitle('Lead-Lag Cross-Correlation: Search Volume Dev vs. Stock Returns', fontsize=16)

    for i, ticker in enumerate(valid_tickers):
        df = data_dict[ticker]

        # Use search_ma_dev as the X variable and return as Y
        search_dev = df['search_ma_dev'].dropna().values
        returns = df['return'].dropna().values

        if len(search_dev) < max_lag * 2 + 1 or len(returns) < max_lag * 2 + 1:
             print(f"Not enough data for {ticker} ({min(len(search_dev), len(returns))} points) "
                   f"for cross-correlation analysis with max_lag={max_lag}.")
             if i < len(axes): # Hide unused subplot if this ticker is skipped
                 axes[i].axis('off')
             continue

        lags, corrs = lead_lag_correlation(search_dev, returns, max_lag=max_lag)

        ax = axes[i]
        colors = ['green' if l > 0 else ('blue' if l == 0 else 'gray') for l in lags]
        ax.bar(lags, corrs, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Significance bounds (approximate 95% CI for correlation)
        n_obs = min(len(search_dev), len(returns))
        if n_obs > 0:
            sig_bound = 1.96 / np.sqrt(n_obs - max_lag) if n_obs > max_lag else 0 # Adjusted for effective degrees of freedom
            ax.axhline(y=sig_bound, color='red', linestyle='--', alpha=0.5, label='95% CI')
            ax.axhline(y=-sig_bound, color='red', linestyle='--', alpha=0.5)
            if i == 0: # Add legend only once
                ax.legend(loc='upper right')

        ax.set_title(f'{ticker}', fontsize=11)
        ax.set_xlabel('Lag (weeks, positive = search leads returns)')
        ax.set_ylabel('Correlation')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig) # Close the figure to free memory
    print(f"Cross-correlation plot saved to {output_path}")

def perform_granger_causality(data_dict: dict[str, pd.DataFrame], tickers: list[str],
                              maxlag: int = DEFAULT_MAX_LAG_GRANGER) -> dict:
    """
    Performs Granger causality tests for search_ma_dev -> return for specified tickers.

    Args:
        data_dict (dict): Dictionary of processed data for each ticker.
        tickers (list): List of tickers to test.
        maxlag (int): Maximum number of lags to test for Granger causality.

    Returns:
        dict: A dictionary containing Granger causality test results (p-values) for each ticker and lag.
    """
    granger_results = {}
    for ticker in tickers:
        if ticker not in data_dict:
            continue
        df = data_dict[ticker][['return', 'search_ma_dev']].dropna()

        print(f"\n{'='*50}")
        print(f"Granger Causality Test: {ticker}")
        print(f"H0: Search volume (search_ma_dev) does NOT Granger-cause returns")
        print(f"{'='*50}")

        if len(df) < maxlag + 2: # Need at least maxlag + 1 observations for the test to run
            print(f"Not enough observations ({len(df)}) for {ticker} to perform Granger Causality with maxlag={maxlag}.")
            granger_results[ticker] = "Not enough data"
            continue

        try:
            # grangercausalitytests expects the dependent variable (y) first, then the independent variable (x)
            # So, we pass [['return', 'search_ma_dev']] for 'search_ma_dev' Granger-causing 'return'
            results = grangercausalitytests(df[['return', 'search_ma_dev']], maxlag=maxlag, verbose=False)
            ticker_results = {}
            for lag in range(1, maxlag + 1):
                if lag in results:
                    f_pval = results[lag][0]['ssr_ftest'][1]
                    print(f"Lag {lag}: F-test p-value = {f_pval:.4f}", end=" ")
                    if f_pval < 0.01:
                        print(f"*** (Significant at 1%)")
                    elif f_pval < 0.05:
                        print(f"** (Significant at 5%)")
                    elif f_pval < 0.10:
                        print(f"* (Significant at 10%)")
                    else:
                        print(f"")
                    ticker_results[f'lag_{lag}_pval'] = f_pval
                else:
                    print(f"Lag {lag}: Test results not available.")
                    ticker_results[f'lag_{lag}_pval'] = np.nan
            granger_results[ticker] = ticker_results
        except Exception as e:
            print(f"Error performing Granger Causality for {ticker}: {e}")
            granger_results[ticker] = f"Error: {e}"
    return granger_results

def alt_data_signal(df: pd.DataFrame, ma_window: int = DEFAULT_MA_WINDOW) -> pd.DataFrame:
    """
    Constructs a binary investment signal from Google Trends data.
    Signal = 1 (bullish) when search volume > MA, 0 (neutral/bearish) when search volume <= MA.
    Calculates strategy and benchmark returns.

    Args:
        df (pd.DataFrame): Processed data for a single ticker, including 'search_volume' and 'return'.
        ma_window (int): Moving average window for signal generation.

    Returns:
        pd.DataFrame: DataFrame with 'signal', 'strategy_return', and 'benchmark_return' columns added.
    """
    df_copy = df.copy()

    if 'search_volume' not in df_copy.columns or 'return' not in df_copy.columns:
        raise ValueError("DataFrame must contain 'search_volume' and 'return' columns.")

    # Ensure search_ma is calculated before signal
    df_copy['search_ma'] = df_copy['search_volume'].rolling(window=ma_window).mean()

    # Generate the signal: buy when search volume is above its moving average
    df_copy['signal'] = (df_copy['search_volume'] > df_copy['search_ma']).astype(int)

    # Position: long when signal=1, flat when signal=0. Shift signal to apply to next period's return.
    # A signal generated at time t will influence the decision for returns at time t+1.
    df_copy['strategy_return'] = df_copy['signal'].shift(1) * df_copy['return']
    df_copy['benchmark_return'] = df_copy['return'] # Buy-and-hold benchmark

    return df_copy.dropna()

def calculate_rolling_ic(signal_values: pd.Series, forward_returns: pd.Series,
                         window: int = DEFAULT_IC_WINDOW) -> pd.Series:
    """
    Calculates rolling Information Coefficient (IC) over a specified window.

    Args:
        signal_values (pd.Series): Series of signal values (e.g., binary signal).
        forward_returns (pd.Series): Series of forward returns (returns at t+1).
        window (int): Rolling window size (e.g., 104 for 2 years of weekly data).

    Returns:
        pd.Series: Series of rolling IC values, indexed by the end date of the window.
                   Returns an empty Series if there isn't enough data.
    """
    ic_values = []
    # Ensure indices are aligned and drop any NaNs from the merge
    aligned_data = pd.DataFrame({'signal': signal_values, 'forward_returns': forward_returns}).dropna()

    if len(aligned_data) < window:
        return pd.Series([], dtype=float) # Return empty series if not enough data for at least one window

    for i in range(len(aligned_data) - window + 1):
        subset = aligned_data.iloc[i : i + window]
        if len(subset) > 1: # Spearmanr requires at least 2 observations
            ic, _ = spearmanr(subset['signal'], subset['forward_returns'])
            ic_values.append(ic)
        else:
            ic_values.append(np.nan)
    return pd.Series(ic_values, index=aligned_data.index[window-1:])


def evaluate_signal_performance(data_dict: dict[str, pd.DataFrame], tickers: list[str],
                                ma_window: int = DEFAULT_MA_WINDOW,
                                annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
                                ic_window: int = DEFAULT_IC_WINDOW,
                                output_prefix: str = "signal_performance") -> pd.DataFrame:
    """
    Evaluates the performance of the alternative data signal for each ticker.
    Calculates Sharpe Ratio, Cumulative Return, Hit Rate, Average IC, ICIR, and plots results.

    Args:
        data_dict (dict): Dictionary of processed data for each ticker.
        tickers (list): List of tickers to evaluate.
        ma_window (int): Moving average window for signal generation.
        annualization_factor (int): Factor to annualize returns (e.g., 52 for weekly).
        ic_window (int): Window for rolling IC calculation.
        output_prefix (str): Prefix for saving plot files.

    Returns:
        pd.DataFrame: A summary DataFrame of performance metrics. Returns empty DataFrame if no data.
    """
    summary = {}

    valid_tickers = [t for t in tickers if t in data_dict and not data_dict[t].empty]
    if not valid_tickers:
        print("No valid tickers with processed data found for signal evaluation.")
        return pd.DataFrame()

    num_plots = len(valid_tickers)
    
    # Initialize figures for various plots
    fig_cum_ret, axes_cum_ret = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))
    fig_ic_decay, axes_ic_decay = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))
    fig_scatter, axes_scatter = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))

    # Ensure axes are always iterable, even for a single subplot
    if num_plots == 1:
        axes_cum_ret = [axes_cum_ret]
        axes_ic_decay = [axes_ic_decay]
        axes_scatter = [axes_scatter]

    for i, ticker in enumerate(valid_tickers):
        df_processed = data_dict[ticker].copy()
        try:
            df = alt_data_signal(df_processed, ma_window=ma_window)
        except ValueError as e:
            print(f"Skipping {ticker} signal generation due to error: {e}")
            continue

        strat_ret = df['strategy_return'].dropna()
        bench_ret = df['benchmark_return'].dropna()
        signal_data = df['signal'].dropna()

        # Ensure there's enough data for meaningful calculations
        if strat_ret.empty or len(strat_ret) < annualization_factor:
            print(f"Skipping {ticker} evaluation: Not enough data for returns after signal generation.")
            # Ensure plots don't show empty subplots if a ticker is skipped
            if i < len(axes_cum_ret): axes_cum_ret[i].axis('off')
            if i < len(axes_ic_decay): axes_ic_decay[i].axis('off')
            if i < len(axes_scatter): axes_scatter[i].axis('off')
            continue

        # Performance Metrics
        signal_sharpe = (strat_ret.mean() * annualization_factor) / (strat_ret.std() * np.sqrt(annualization_factor)) if strat_ret.std() != 0 else np.nan
        bnh_sharpe = (bench_ret.mean() * annualization_factor) / (bench_ret.std() * np.sqrt(annualization_factor)) if bench_ret.std() != 0 else np.nan

        strat_cum_ret = (1 + strat_ret).cumprod().iloc[-1] - 1 if not strat_ret.empty else np.nan
        bnh_cum_ret = (1 + bench_ret).cumprod().iloc[-1] - 1 if not bench_ret.empty else np.nan

        hit_rate = (strat_ret > 0).mean() if not strat_ret.empty else np.nan
        pct_time_invested = signal_data.mean() if not signal_data.empty else np.nan

        # Information Coefficient (IC) - Correlate signal at t with return at t+1
        ic_df = df[['signal', 'return']].copy()
        avg_ic, icir = np.nan, np.nan
        rolling_ic = pd.Series([], dtype=float)

        if not ic_df.empty:
            ic_df['forward_return'] = ic_df['return'].shift(-1)
            ic_df = ic_df.dropna() # Drop rows where forward_return is NaN (last row)
            
            if len(ic_df) > 1: # Need at least 2 points for spearmanr
                avg_ic, _ = spearmanr(ic_df['signal'], ic_df['forward_return'])
                rolling_ic = calculate_rolling_ic(ic_df['signal'], ic_df['forward_return'], window=ic_window)
                std_ic = rolling_ic.std()
                icir = avg_ic / std_ic if std_ic != 0 else np.nan
        
        summary[ticker] = {
            'Signal Sharpe': signal_sharpe,
            'B&H Sharpe': bnh_sharpe,
            'Signal Cum Return': strat_cum_ret,
            'B&H Cum Return': bnh_cum_ret,
            'Hit Rate': hit_rate,
            'Pct Time Invested': pct_time_invested,
            'Avg IC': avg_ic,
            'ICIR': icir
        }

        # Plot Cumulative Returns
        ax_cr = axes_cum_ret[i]
        (1 + strat_ret).cumprod().plot(ax=ax_cr, label='Signal Strategy', color='green')
        (1 + bench_ret).cumprod().plot(ax=ax_cr, label='Buy & Hold', color='red', linestyle='--')
        ax_cr.set_title(f'{ticker} - Cumulative Returns', fontsize=12)
        ax_cr.set_ylabel('Cumulative Return')
        ax_cr.legend()
        ax_cr.grid(True, linestyle='--', alpha=0.6)

        # Plot Rolling IC (Signal Decay)
        ax_ic = axes_ic_decay[i]
        if not rolling_ic.empty and len(rolling_ic) > 1:
            rolling_ic.plot(ax=ax_ic, title=f'{ticker} - Rolling {round(ic_window/annualization_factor)}-Year IC (Signal Decay)', color='purple')
            ax_ic.axhline(0.05, color='gray', linestyle=':', label='IC > 0.05 (Meaningful)')
            ax_ic.axhline(0.10, color='gray', linestyle='--', label='IC > 0.10 (Strong)')
            ax_ic.set_ylabel('Rolling IC')
            ax_ic.legend()
        else:
            ax_ic.text(0.5, 0.5, "Not enough data for rolling IC", horizontalalignment='center', verticalalignment='center', transform=ax_ic.transAxes)
        ax_ic.grid(True, linestyle='--', alpha=0.6)

        # Scatter Plot: Search MA Dev vs. Next-Week Return
        ax_sc = axes_scatter[i]
        scatter_df = df_processed[['search_ma_dev', 'return']].copy().dropna()
        if not scatter_df.empty:
            scatter_df['next_week_return'] = scatter_df['return'].shift(-1)
            scatter_df = scatter_df.dropna()
            if not scatter_df.empty and len(scatter_df) > 1:
                sns.regplot(x='search_ma_dev', y='next_week_return', data=scatter_df, ax=ax_sc, scatter_kws={'alpha':0.3})
                corr_val = scatter_df['search_ma_dev'].corr(scatter_df['next_week_return'])
                ax_sc.set_title(f'{ticker} - Search MA Dev vs. Next-Week Return (R²={corr_val**2:.2f})', fontsize=12)
                ax_sc.set_xlabel('Search Moving Average Deviation')
                ax_sc.set_ylabel('Next-Week Return')
            else:
                ax_sc.text(0.5, 0.5, "Not enough data for scatter plot", horizontalalignment='center', verticalalignment='center', transform=ax_sc.transAxes)
        else:
            ax_sc.text(0.5, 0.5, "Not enough data for scatter plot", horizontalalignment='center', verticalalignment='center', transform=ax_sc.transAxes)

    fig_cum_ret.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_ic_decay.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_scatter.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_cum_ret.savefig(f'{output_prefix}_cumulative_returns_comparison.png', dpi=150)
    fig_ic_decay.savefig(f'{output_prefix}_rolling_ic_decay.png', dpi=150)
    fig_scatter.savefig(f'{output_prefix}_search_dev_vs_next_return_scatter.png', dpi=150)

    # Close figures to free memory
    plt.close(fig_cum_ret)
    plt.close(fig_ic_decay)
    plt.close(fig_scatter)
    
    print(f"Signal performance plots saved with prefix '{output_prefix}'.")

    return pd.DataFrame(summary).T.round(3)

def evaluate_alt_data_scorecard(scorecard_data: dict, weights: dict) -> float:
    """
    Calculates a weighted overall score for an alternative data source based on a scorecard.

    Args:
        scorecard_data (dict): A dictionary containing dimensions and their scores (e.g., ALT_DATA_EVALUATION).
        weights (dict): A dictionary containing weights for each dimension (e.g., ALT_DATA_WEIGHTS).

    Returns:
        float: The weighted overall score.
    """
    overall_score = sum(scorecard_data.get(k, 0) * weights.get(k, 0) for k in weights.keys())
    return overall_score

# --- Main Orchestration Function for the Application ---

def analyze_google_trends_alt_data(
    queries: dict = DEFAULT_QUERIES,
    tickers: list = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    interval: str = DEFAULT_INTERVAL,
    ma_window: int = DEFAULT_MA_WINDOW,
    max_lag_corr: int = DEFAULT_MAX_LAG_CORR,
    max_lag_granger: int = DEFAULT_MAX_LAG_GRANGER,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    ic_window: int = DEFAULT_IC_WINDOW,
    plot_output_dir: str = '.',
    evaluate_scorecard: bool = True
) -> tuple[dict, pd.DataFrame, dict, dict, pd.DataFrame, float | None]:
    """
    Orchestrates the entire alternative data analysis workflow for Google Trends.

    Args:
        queries (dict): A dictionary where keys are tickers and values are lists of keywords.
        tickers (list): List of tickers to analyze. If None, uses keys from 'queries'.
        start_date (str): Start date for data retrieval (YYYY-MM-DD).
        end_date (str): End date for data retrieval (YYYY-MM-DD).
        interval (str): Data interval (e.g., '1wk').
        ma_window (int): Moving average window size for preprocessing and signal generation.
        max_lag_corr (int): Maximum lag for cross-correlation plotting.
        max_lag_granger (int): Maximum lags for Granger causality test.
        annualization_factor (int): Factor to annualize returns for Sharpe Ratio.
        ic_window (int): Window for rolling IC calculation.
        plot_output_dir (str): Directory to save generated plots.
        evaluate_scorecard (bool): Whether to perform and print the alt data scorecard evaluation.

    Returns:
        tuple: A tuple containing:
            - dict: Raw Google Trends data.
            - pd.DataFrame: Raw financial data.
            - dict: Processed data.
            - dict: Granger causality results.
            - pd.DataFrame: Signal performance summary.
            - float | None: Overall alternative data score (if evaluate_scorecard is True, else None).
    """
    if tickers is None:
        tickers = list(queries.keys())

    os.makedirs(plot_output_dir, exist_ok=True) # Ensure output directory exists

    print("\n--- Step 1: Data Acquisition ---")
    pytrends_client = initialize_pytrends()
    trends_raw = get_google_trends_data(pytrends_client, queries, start_date, end_date)
    financial_raw = get_financial_data(tickers, start_date, end_date, interval)

    print(f"\nTrends data retrieved for {len(trends_raw)} companies.")
    if trends_raw and list(trends_raw.keys()):
        first_ticker = list(trends_raw.keys())[0]
        print(f"\nSample {first_ticker} Google Trends data:")
        print(trends_raw[first_ticker].head())
    if isinstance(financial_raw, pd.DataFrame) and not financial_raw.empty:
        print("\nSample financial data:")
        print(financial_raw.head())

    print("\n--- Step 2: Data Preprocessing ---")
    processed_data = preprocess_data(trends_raw, financial_raw, ma_window=ma_window)

    if processed_data and list(processed_data.keys()):
        first_ticker = list(processed_data.keys())[0]
        print(f"\nSample {first_ticker} processed data with engineered features:")
        print(processed_data[first_ticker].head())
        print(f"\nDescriptive statistics for {first_ticker} search_z:")
        print(processed_data[first_ticker]['search_z'].describe())
    else:
        print("\nNo processed data available for analysis. Exiting early.")
        return trends_raw, financial_raw, processed_data, {}, pd.DataFrame(), None # Early exit

    print("\n--- Step 3: Predictive Power Analysis ---")
    plot_cross_correlations(processed_data, tickers, max_lag=max_lag_corr,
                            output_path=os.path.join(plot_output_dir, 'lead_lag_cross_correlation.png'))

    granger_causality_results = perform_granger_causality(processed_data, tickers, maxlag=max_lag_granger)
    print("\nGranger Causality Results Summary:")
    for ticker, res in granger_causality_results.items():
        if isinstance(res, dict):
            # Print average p-value or just a summary of significant lags
            significant_lags = [f"lag_{k.split('_')[1]}" for k, v in res.items() if v < 0.10]
            if significant_lags:
                print(f"  {ticker}: Significant at {', '.join(significant_lags)} (p<0.10)")
            else:
                print(f"  {ticker}: No significant Granger causality found (p<0.10)")
        else:
            print(f"  {ticker}: {res}")


    print("\n--- Step 4: Signal Generation and Performance Evaluation ---")
    signal_performance_summary = evaluate_signal_performance(
        processed_data, tickers, ma_window=ma_window,
        annualization_factor=annualization_factor, ic_window=ic_window,
        output_prefix=os.path.join(plot_output_dir, 'signal_performance')
    )
    print("\n--- Signal Performance Summary ---")
    print(signal_performance_summary)

    overall_alt_data_score = None
    if evaluate_scorecard:
        print("\n--- Step 5: Alternative Data Evaluation Scorecard ---")
        overall_alt_data_score = evaluate_alt_data_scorecard(ALT_DATA_EVALUATION, ALT_DATA_WEIGHTS)
        print("\n--- Alternative Data Evaluation Scorecard for Google Trends ---")
        for dimension, score in ALT_DATA_EVALUATION.items():
            if dimension != 'Dataset': # Skip printing 'Dataset' as a dimension
                print(f"{dimension:<20}: {score}")
        print(f"\nOverall Alt Data Score: {overall_alt_data_score:.1f} / 5.0")

    return (trends_raw, financial_raw, processed_data,
            granger_causality_results, signal_performance_summary,
            overall_alt_data_score)

# --- Main execution block for when the script is run directly (for testing/demonstration) ---
if __name__ == "__main__":
    print("Starting Google Trends Alternative Data Analysis...")

    # Customize parameters here for a specific run
    # For demonstration, using slightly adjusted dates for faster execution
    
    # Ensure the plots directory exists
    PLOT_DIR = './alt_data_plots'
    os.makedirs(PLOT_DIR, exist_ok=True)

    (raw_trends_data, raw_financial_data, processed_analysis_data,
     granger_results_summary, performance_table, final_score) = \
        analyze_google_trends_alt_data(
            queries=DEFAULT_QUERIES,
            tickers=DEFAULT_TICKERS,
            start_date='2020-01-01', # Shorter range for demo
            end_date='2023-12-31',
            plot_output_dir=PLOT_DIR
        )

    print("\n--- Analysis Complete ---")
    print("\nRaw Trends Data keys available:", list(raw_trends_data.keys()))
    print("Raw Financial Data columns available:", raw_financial_data.columns.tolist() if not raw_financial_data.empty else "N/A")
    print("Processed Data keys available:", list(processed_analysis_data.keys()))
    print("Granger Causality Results (partial view):\n", {k: v for k, v in granger_results_summary.items() if isinstance(v, dict)})
    print("Signal Performance Summary Table:\n", performance_table)
    if final_score is not None:
        print(f"Final Overall Alt Data Score: {final_score:.1f}")

    # You can now use the returned dataframes and dictionaries in further analysis or display in an app.
