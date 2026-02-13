
# Streamlit Application Specification: Google Trends for Retail Revenue & Stock Performance Prediction

## Application Overview

*Purpose:* This Streamlit application serves as a comprehensive tool for CFA Charterholders and Investment Professionals to explore and evaluate Google Trends data as an alternative data source for predicting consumer-facing company performance. It guides the user through a real-world workflow, from data acquisition and preprocessing to statistical validation, signal construction, and a structured evaluation of the alternative data. The application simulates the process an equity analyst, like Sarah Chen, would follow to discover and assess new informational edges.

*High-level Story Flow:*

1.  **Introduction**: The application sets the stage, introducing the persona of Sarah Chen, a CFA equity analyst, and the scenario of leveraging Google Trends for alpha generation. It highlights the importance of alternative data in modern finance.
2.  **Data Acquisition**: Users define their research scope by selecting a consumer-facing company, specifying relevant Google search terms, and setting a historical date range. The app then fetches raw Google Trends search interest data and historical stock prices using external APIs.
3.  **Data Preprocessing & Feature Engineering**: The acquired raw data undergoes critical preprocessing steps: cleaning, resampling to a consistent weekly frequency, date alignment, and normalization of Google Trends data to z-scores. Key predictive features like week-over-week changes and moving average deviations are engineered.
4.  **Exploratory Visual Analysis**: Initial hypotheses are formed and visually checked by plotting dual-axis time-series charts, comparing normalized Google Trends index with the cumulative stock return for the selected company. This step aims to uncover intuitive lead-lag patterns.
5.  **Lead-Lag Cross-Correlation Analysis**: Formal quantification of the lead-lag relationship is performed. Bar charts display cross-correlation values at various lags, indicating whether search activity leads or lags stock returns, alongside statistical significance bounds.
6.  **Granger Causality Testing**: To establish statistical predictive power, Granger causality tests are conducted. This step assesses whether past Google Trends data significantly improves forecasts of future stock returns beyond what past returns alone provide.
7.  **Prototype Signal Construction & Performance Evaluation**: A simple prototype investment signal is constructed based on search volume deviations from its moving average. The signal's performance is then rigorously evaluated against a buy-and-hold benchmark using metrics such as Sharpe Ratio, Cumulative Return, Hit Rate, Information Coefficient (IC), and IC Information Ratio (ICIR). Signal decay analysis is also presented.
8.  **Alternative Data Evaluation Framework**: Google Trends is assessed against a structured 7-dimension scorecard (Predictive Power, Uniqueness, Coverage, Timeliness, History, Legality, Cost). This holistic evaluation provides a comprehensive understanding of Google Trends' strengths, limitations, and risks as an alternative data source, crucial for responsible integration into investment strategies.

## Code Requirements

### Imports

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io # To capture stdout for Granger causality
import sys # To capture stdout for Granger causality
import os # For image paths

# Import all functions and global variables from source.py
# This includes get_google_trends_data, get_financial_data, preprocess_data,
# plot_cross_correlations, perform_granger_causality, evaluate_signal_performance,
# evaluate_alt_data_scorecard, along with 'queries', 'evaluation', 'weights' dictionaries, etc.
from source import * 
```

### `st.session_state` Design

The `st.session_state` is utilized to preserve the application's state across user interactions and page navigations, preventing redundant API calls and computations.

**Initialization (at the start of `app.py`):**

```python
# User Inputs and Global Settings
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'NKE' # Default
if 'search_term' not in st.session_state:
    # Use the 'queries' dictionary from source.py for default
    st.session_state.search_term = queries[st.session_state.selected_ticker][0] 
if 'start_date' not in st.session_state:
    st.session_state.start_date = pd.to_datetime('2019-01-01').date()
if 'end_date' not in st.session_state:
    st.session_state.end_date = pd.to_datetime('2024-01-01').date()
if 'ma_window' not in st.session_state:
    st.session_state.ma_window = 12 # Default for moving average
if 'max_lag_corr' not in st.session_state:
    st.session_state.max_lag_corr = 8 # Default for cross-correlation lags
if 'granger_maxlag' not in st.session_state:
    st.session_state.granger_maxlag = 4 # Default for Granger causality lags
if 'annualization_factor' not in st.session_state:
    st.session_state.annualization_factor = 52 # For weekly data, used in performance metrics
if 'ic_window' not in st.session_state:
    st.session_state.ic_window = 104 # 2-year rolling window for weekly data IC calculation

# Data and Results Storage
if 'trends_raw_data' not in st.session_state:
    st.session_state.trends_raw_data = {} # Stores output from get_google_trends_data
if 'financial_raw_data' not in st.session_state:
    st.session_state.financial_raw_data = pd.DataFrame() # Stores output from get_financial_data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {} # Stores output from preprocess_data
if 'granger_results_output' not in st.session_state:
    st.session_state.granger_results_output = "" # Stores captured stdout from perform_granger_causality
if 'signal_performance_summary' not in st.session_state:
    st.session_state.signal_performance_summary = pd.DataFrame() # Stores output from evaluate_signal_performance
if 'overall_alt_data_score' not in st.session_state:
    st.session_state.overall_alt_data_score = None # Stores output from evaluate_alt_data_scorecard

# Temporary flag to indicate if plots have been generated (for image display)
if 'plots_generated_exploratory' not in st.session_state:
    st.session_state.plots_generated_exploratory = False
if 'plots_generated_correlation' not in st.session_state:
    st.session_state.plots_generated_correlation = False
if 'plots_generated_signal' not in st.session_state:
    st.session_state.plots_generated_signal = False
```

**Update and Read across pages:**

*   **User Inputs**: `selected_ticker`, `search_term`, `start_date`, `end_date`, `ma_window`, `max_lag_corr`, `granger_maxlag`, `annualization_factor`, and `ic_window` are updated by user widgets and read by various `source.py` functions when invoked.
*   **Intermediate Data**:
    *   `trends_raw_data` and `financial_raw_data` are updated after "Data Acquisition" (Page 1) and read by "Data Preprocessing & Feature Engineering" (Page 2).
    *   `processed_data` is updated after "Data Preprocessing & Feature Engineering" (Page 2) and read by "Exploratory Visual Analysis" (Page 3), "Lead-Lag Cross-Correlation Analysis" (Page 4), "Granger Causality Testing" (Page 5), and "Prototype Signal Construction & Performance Evaluation" (Page 6).
*   **Results**:
    *   `granger_results_output` is updated after "Granger Causality Testing" (Page 5) and displayed on that page.
    *   `signal_performance_summary` is updated after "Prototype Signal Construction & Performance Evaluation" (Page 6) and displayed on that page.
    *   `overall_alt_data_score` is updated after "Alternative Data Evaluation Framework" (Page 7) and displayed on that page.
*   **Plot Flags**: `plots_generated_exploratory`, `plots_generated_correlation`, `plots_generated_signal` are set to `True` when corresponding plotting functions are called, allowing subsequent page renders to display the saved images.

### UI Interactions and `source.py` Function Calls

The application simulates a multi-page experience using a `st.sidebar.selectbox` for navigation.

**Sidebar Navigation:**

```python
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
```

#### Page: "Introduction"

*   **Markdown:**
    ```python
    st.markdown(f"## Alternative Data Signals: Google Trends for Retail Revenue & Stock Performance Prediction")
    st.markdown(f"**Persona: Sarah Chen, CFA, Equity Analyst at Alpha Insights**")
    st.markdown(f"**Scenario:**")
    st.markdown(f"As an equity analyst at *Alpha Insights*, a quantitative asset management firm, I'm constantly seeking new informational edges to generate alpha for our portfolios. Traditional financial data, while essential, often reflects information already priced into the market. My current focus is on alternative data—specifically, public search interest captured by Google Trends—to see if it can provide an early indicator of consumer demand shifts and, consequently, anticipate earnings surprises or movements in stock prices for consumer-facing companies.")
    st.markdown(f"Today, I'm examining a leading consumer brand, **Nike ($NKE)**, to investigate whether surges in search interest for their products precede increases in sales or stock returns. This real-world workflow will take me from raw data acquisition through statistical validation to the construction of a prototype signal, and finally, a structured evaluation of the alternative data source itself. This systematic approach is critical for incorporating non-traditional data responsibly into our investment strategies.")
    st.markdown(f"Use the sidebar to navigate through the workflow.")
    ```

#### Page: "1. Data Acquisition"

*   **Markdown:**
    ```python
    st.markdown(f"## 1. Setup & Data Acquisition: Gathering the Raw Materials")
    st.markdown(f"As an analyst at *Alpha Insights*, my first step is always to gather the necessary data. For this investigation, I need two primary sources: historical Google Trends search interest for brand-related terms and the brand's historical stock prices. The challenge with alternative data often begins with sourcing—ensuring I retrieve comprehensive and relevant data while adhering to API best practices.")
    ```
*   **Widgets:**
    *   `st.session_state.selected_ticker` is updated via `st.selectbox`.
        ```python
        available_tickers = list(queries.keys()) # 'queries' is from source.py
        st.session_state.selected_ticker = st.selectbox(
            "Select a Company Ticker:",
            options=available_tickers,
            index=available_tickers.index(st.session_state.selected_ticker)
        )
        ```
    *   `st.session_state.search_term` is updated via `st.selectbox`, dynamically filtered based on `selected_ticker`.
        ```python
        available_search_terms = queries[st.session_state.selected_ticker]
        st.session_state.search_term = st.selectbox(
            f"Select a Search Term for {st.session_state.selected_ticker}:",
            options=available_search_terms,
            index=available_search_terms.index(st.session_state.search_term) if st.session_state.search_term in available_search_terms else 0
        )
        ```
    *   `st.session_state.start_date` is updated via `st.date_input`.
        ```python
        st.session_state.start_date = st.date_input("Start Date:", st.session_state.start_date)
        ```
    *   `st.session_state.end_date` is updated via `st.date_input`.
        ```python
        st.session_state.end_date = st.date_input("End Date:", st.session_state.end_date)
        ```
    *   `st.button("Fetch Data")` triggers data acquisition.
*   **Function Calls (on button click):**
    *   `queries_dict_for_trends = {st.session_state.selected_ticker: [st.session_state.search_term]}`
    *   `st.session_state.trends_raw_data = get_google_trends_data(queries_dict=queries_dict_for_trends, start_date=st.session_state.start_date.isoformat(), end_date=st.session_state.end_date.isoformat())`
    *   `st.session_state.financial_raw_data = get_financial_data(tickers=[st.session_state.selected_ticker], start_date=st.session_state.start_date.isoformat(), end_date=st.session_state.end_date.isoformat(), interval='1wk')`
*   **Session State Update**: `st.session_state.trends_raw_data`, `st.session_state.financial_raw_data`.
*   **Markdown:**
    ```python
    st.markdown(f"The raw Google Trends data provides search interest as a relative index, not absolute volume. This means the values (0-100) are normalized to the peak interest within the queried time window, and they can vary slightly with repeated API calls due to sampling. This is a common characteristic of free alternative data—it requires careful preprocessing. The financial data, on the other hand, provides clean historical adjusted close prices.")
    st.markdown(f"---")
    st.markdown(f"### Practitioner Warning:")
    st.markdown(f"Google Trends data is noisy and non-reproducible. Repeated API calls for the same query can return slightly different values because Google samples from its full search database. The index is also relative to the query window—changing the start/end date changes all values. Best practices: (a) always retrieve the full desired window in a single call, (b) average multiple calls to reduce sampling noise, (c) document the exact retrieval date and parameters. For production use, institutional investors subscribe to vendors (e.g., Quandl/Nasdaq Data Link) that provide cleaned, versioned Google Trends data. For this teaching exercise, the raw pytrends output is sufficient.")
    ```
*   **Display (if data fetched):**
    ```python
    if st.session_state.selected_ticker in st.session_state.trends_raw_data:
        st.markdown(f"### Sample {st.session_state.selected_ticker} Google Trends Data:")
        st.dataframe(st.session_state.trends_raw_data[st.session_state.selected_ticker].head())
    if not st.session_state.financial_raw_data.empty:
        st.markdown(f"### Sample {st.session_state.selected_ticker} Financial Data:")
        st.dataframe(st.session_state.financial_raw_data[[st.session_state.selected_ticker]].head()) # Display only selected ticker
    ```

#### Page: "2. Data Preprocessing & Feature Engineering"

*   **Pre-conditions**: Requires `st.session_state.trends_raw_data` and `st.session_state.financial_raw_data` to be populated. Display error if not.
*   **Markdown:**
    ```python
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
    ```
*   **Widgets:**
    *   `st.session_state.ma_window` is updated via `st.number_input`.
        ```python
        st.session_state.ma_window = st.number_input("Moving Average Window (weeks):", min_value=4, max_value=52, value=st.session_state.ma_window)
        ```
    *   `st.button("Process Data")` triggers preprocessing.
*   **Function Calls (on button click):**
    *   `st.session_state.processed_data = preprocess_data(trends_dict=st.session_state.trends_raw_data, financial_df=st.session_state.financial_raw_data, ma_window=st.session_state.ma_window)`
*   **Session State Update**: `st.session_state.processed_data`.
*   **Markdown:**
    ```python
    st.markdown(f"Z-scoring allows me to compare search interest trends across different companies, even if their absolute search volumes vary greatly. The moving average deviation, in particular, helps identify when search interest is unusually high or low relative to its recent trend. This kind of deviation could signal a shift in consumer attention that might precede changes in financial performance, making it a potentially strong predictive feature for my models at *Alpha Insights*.")
    ```
*   **Display (if data processed):**
    ```python
    if st.session_state.selected_ticker in st.session_state.processed_data:
        st.markdown(f"### Sample {st.session_state.selected_ticker} Processed Data with Engineered Features:")
        st.dataframe(st.session_state.processed_data[st.session_state.selected_ticker].head())
        st.markdown(f"### Descriptive Statistics for {st.session_state.selected_ticker} Z-scored Search Volume:")
        st.dataframe(st.session_state.processed_data[st.session_state.selected_ticker]['search_z'].describe())
    ```

#### Page: "3. Exploratory Visual Analysis"

*   **Pre-conditions**: Requires `st.session_state.processed_data` to be populated. Display error if not.
*   **Markdown:**
    ```python
    st.markdown(f"## 3. Exploratory Visual Analysis: Spotting Initial Patterns")
    st.markdown(f"Before diving into complex statistical models, I always start with visual exploration. As an equity analyst, I'm looking for intuitive lead-lag patterns—do spikes in '{st.session_state.selected_ticker}' search interest visually precede upticks in its stock performance? This helps me form initial hypotheses and sanity-check the data before more rigorous analysis. I'll create dual-axis time-series charts, plotting the normalized Google Trends index against the cumulative stock return for each company.")
    ```
*   **Widgets:**
    *   `st.button("Generate Dual-Axis Time-Series Chart")` triggers plot generation.
*   **Function Calls (on button click):**
    *   **Assumption:** A function `plot_dual_axis_trends_vs_returns` exists in `source.py` which replicates the logic from "3. Exploratory Visual Analysis" markdown code cell in the original notebook, taking `processed_data` and a list of `tickers` and saving the plot to `trends_vs_returns.png`.
    *   `plot_dual_axis_trends_vs_returns(data_dict=st.session_state.processed_data, tickers=[st.session_state.selected_ticker])`
    *   `st.session_state.plots_generated_exploratory = True`
*   **Session State Update**: `st.session_state.plots_generated_exploratory`.
*   **Markdown:**
    ```python
    st.markdown(f"For consumer-facing companies like {st.session_state.selected_ticker}, I often observe that strong surges in search interest, especially when normalized, visually precede periods of positive stock performance. This informal observation provides initial support for my hypothesis, but it requires rigorous statistical validation. These charts are invaluable for quickly grasping potential relationships and identifying periods where the correlation might be strongest or weakest.")
    ```
*   **Display (if plot generated):**
    ```python
    if st.session_state.plots_generated_exploratory and os.path.exists('trends_vs_returns.png'):
        st.image('trends_vs_returns.png', caption=f'Google Trends Search Volume vs. Stock Performance for {st.session_state.selected_ticker}')
    ```

#### Page: "4. Lead-Lag Cross-Correlation Analysis"

*   **Pre-conditions**: Requires `st.session_state.processed_data` to be populated. Display error if not.
*   **Markdown:**
    ```python
    st.markdown(f"## 4. Lead-Lag Cross-Correlation Analysis: Quantifying Relationships")
    st.markdown(f"Visual inspection is a good starting point, but as an analyst at *Alpha Insights*, I need to quantify these lead-lag relationships formally. The cross-correlation function (CCF) measures the similarity between two time series, '{st.session_state.selected_ticker}' search volume ($x$) and '{st.session_state.selected_ticker}' stock returns ($y$), as a function of the lag applied to one of them. Specifically, I want to see if search volume at time $t$ predicts stock returns at a future time $t+k$.")
    st.markdown(r"The cross-correlation at lag $k$, denoted $P_{{xy}}(k)$, is calculated as:")
    st.markdown(r"$$P_{{xy}}(k) = \frac{{\sum_{{t=1}}^{{T-k}} (x_t - \bar{{x}}) (y_{{t+k}} - \bar{{y}})}}{{\sqrt{{\sum_{{t=1}}^{{T}} (x_t - \bar{{x}})^2 \sum_{{t=1}}^{{T}} (y_t - \bar{{y}})^2}}}}$$")
    st.markdown(r"where $x_t$ is the search volume (or its deviation) at time $t$, $y_t$ is the stock return at time $t$, $\bar{{x}}$ and $\bar{{y}}$ are their respective means, and $T$ is the number of observations.")
    st.markdown(r"*   If $k > 0$, it means $x$ (search volume) leads $y$ (returns) by $k$ periods. I'm particularly interested in positive lags, as they suggest predictive power.")
    st.markdown(r"*   If $k < 0$, it means $x$ lags $y$ by $|k|$ periods.")
    st.markdown(r"*   If $k = 0$, it's the simultaneous correlation.")
    st.markdown(r"To determine if a correlation is statistically meaningful, I compare it against significance bounds. Under the null hypothesis of no correlation, the approximate 95% confidence interval is $\pm 1.96 / \sqrt{{T}}$, where $T$ is the number of observations. Correlations outside these bounds are statistically significant at the 5% level, suggesting a true relationship rather than random noise. For weekly data over 5 years (approx. $T=260$ weeks), this bound is roughly $\pm 1.96 / \sqrt{{260}} \approx \pm 0.12$.")
    ```
*   **Widgets:**
    *   `st.session_state.max_lag_corr` is updated via `st.number_input`.
        ```python
        st.session_state.max_lag_corr = st.number_input("Max Lag for Cross-Correlation (weeks):", min_value=1, max_value=24, value=st.session_state.max_lag_corr)
        ```
    *   `st.button("Calculate & Plot Cross-Correlations")` triggers calculation and plotting.
*   **Function Calls (on button click):**
    *   `plot_cross_correlations(data_dict=st.session_state.processed_data, tickers=[st.session_state.selected_ticker], max_lag=st.session_state.max_lag_corr)`
    *   `st.session_state.plots_generated_correlation = True`
*   **Session State Update**: `st.session_state.plots_generated_correlation`.
*   **Markdown:**
    ```python
    st.markdown(f"The bar charts reveal that for companies like {st.session_state.selected_ticker}, search volume often exhibits positive correlation with future stock returns at lags of 1-4 weeks, and these correlations frequently exceed the calculated significance bounds. This suggests that public interest in the brand, as captured by Google Trends, could indeed be an early indicator of market movement, reinforcing my initial hypothesis. For Apple, the correlations might be weaker due to its larger market capitalization and faster information dissemination. This quantitative insight is a critical step in building a data-driven investment thesis.")
    ```
*   **Display (if plot generated):**
    ```python
    if st.session_state.plots_generated_correlation and os.path.exists('lead_lag_cross_correlation.png'):
        st.image('lead_lag_cross_correlation.png', caption=f'Lead-Lag Cross-Correlation for {st.session_state.selected_ticker}')
    ```

#### Page: "5. Granger Causality Testing"

*   **Pre-conditions**: Requires `st.session_state.processed_data` to be populated. Display error if not.
*   **Markdown:**
    ```python
    st.markdown(f"## 5. Granger Causality Testing: Statistical Predictive Power")
    st.markdown(f"Correlation indicates a relationship, but it does not imply that one variable *causes* or *predicts* another in a statistical sense. For *Alpha Insights*, I need stronger evidence of predictive power. The Granger causality test formalizes whether past values of '{st.session_state.selected_ticker}' search volume ($x$) statistically improve forecasts of future stock returns ($y$), beyond what past returns alone provide. This is crucial for evaluating a signal's potential robustness.")
    st.markdown(f"The test compares two models:")
    st.markdown(r"1.  **Restricted Model ($H_0$):** This model assumes that past values of $x$ (search volume) do not help predict $y$ (returns). It only uses past values of $y$:""")
    st.markdown(r"$$y_t = \alpha + \sum_{{j=1}}^{{p}} \phi_j y_{{t-j}} + \epsilon_t$$")
    st.markdown(r"2.  **Unrestricted Model ($H_1$):** This model assumes that past values of $x$ *do* help predict $y$. It includes both past values of $y$ and past values of $x$:""")
    st.markdown(r"$$y_t = \alpha + \sum_{{j=1}}^{{p}} \phi_j y_{{t-j}} + \sum_{{j=1}}^{{p}} \gamma_j x_{{t-j}} + \eta_t$$")
    st.markdown(r"The F-statistic is then computed to compare the sum of squared residuals (SSR) from these two models:")
    st.markdown(r"$$F = \frac{{(SSR_{{restricted}} - SSR_{{unrestricted}})/p}}{{SSR_{{unrestricted}}/(T - 2p - 1)}} \sim F(p, T - 2p - 1)$$")
    st.markdown(r"where $p$ is the number of lags, and $T$ is the number of observations.")
    st.markdown(r"I reject the null hypothesis ($H_0$: search volume does NOT Granger-cause returns) if the F-test p-value is below a chosen significance level (e.g., 0.05). This provides statistical evidence that past search volume has predictive power for future returns. However, it's an *important caveat* that Granger causality in financial time-series can be fragile; a relationship observed in-sample might not hold out-of-sample due to market adaptation or other factors.")
    ```
*   **Widgets:**
    *   `st.session_state.granger_maxlag` is updated via `st.number_input`.
        ```python
        st.session_state.granger_maxlag = st.number_input("Max Lag for Granger Causality Test (weeks):", min_value=1, max_value=8, value=st.session_state.granger_maxlag)
        ```
    *   `st.button("Perform Granger Causality Test")` triggers the test.
*   **Function Calls (on button click):**
    *   Capture `perform_granger_causality`'s printed output.
    *   `buffer = io.StringIO()`
    *   `sys.stdout = buffer`
    *   `perform_granger_causality(data_dict=st.session_state.processed_data, tickers=[st.session_state.selected_ticker], maxlag=st.session_state.granger_maxlag)`
    *   `sys.stdout = sys.__stdout__`
    *   `st.session_state.granger_results_output = buffer.getvalue()`
*   **Session State Update**: `st.session_state.granger_results_output`.
*   **Markdown:**
    ```python
    st.markdown(f"For {st.session_state.selected_ticker}, the Granger causality test often yields p-values below 0.05 for lags of 1-2 weeks when examining `search_ma_dev`'s influence on returns. This suggests that past deviations in search volume indeed help predict future stock returns, providing statistical backing to my hypothesis. For companies like Apple, which are highly efficient and incorporate information quickly, the p-values might be higher, indicating a weaker or non-existent predictive relationship from this specific alternative data source. This statistical validation is crucial for deciding whether to integrate such a signal into our quantitative models at *Alpha Insights*.")
    ```
*   **Display (if results available):**
    ```python
    if st.session_state.granger_results_output:
        st.markdown("### Granger Causality Test Results:")
        st.text(st.session_state.granger_results_output)
    ```

#### Page: "6. Prototype Signal Construction & Performance Evaluation"

*   **Pre-conditions**: Requires `st.session_state.processed_data` to be populated. Display error if not.
*   **Markdown:**
    ```python
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
    ```
*   **Widgets:**
    *   `st.button("Evaluate Signal Performance")` triggers performance evaluation.
*   **Function Calls (on button click):**
    *   `st.session_state.signal_performance_summary = evaluate_signal_performance(data_dict=st.session_state.processed_data, tickers=[st.session_state.selected_ticker], annualization_factor=st.session_state.annualization_factor, ic_window=st.session_state.ic_window)`
    *   `st.session_state.plots_generated_signal = True`
*   **Session State Update**: `st.session_state.signal_performance_summary`, `st.session_state.plots_generated_signal`.
*   **Markdown:**
    ```python
    st.markdown(f"For {st.session_state.selected_ticker}, the prototype signal shows a modest improvement in Sharpe Ratio and a positive hit rate, indicating it correctly predicts positive returns more than 50% of the time. The average IC is positive, suggesting some predictive power, although it might fall into the 'meaningful' rather than 'strong' category. The rolling IC plot helps me identify if the signal's effectiveness is decaying over time, which is common with crowded alternative data sources. The scatter plot further confirms a slight positive relationship between search moving average deviation and next-week stock returns.")
    st.markdown(f"---")
    st.markdown(f"### Practitioner Warning:")
    st.markdown(f"This is a prototype, not a production strategy. A viable alternative data signal for *Alpha Insights* would require multi-year out-of-sample testing, rigorous transaction cost modeling, and, most importantly, combination with other signals in a multi-factor model. Free alternative data often produces marginal signals due to market efficiency and crowding risk. Realistic expectations for Google Trends-based signals include an IC around 0.02-0.05 and a Sharpe improvement of 0.1-0.3 versus buy-and-hold, and results that are statistically significant for some companies but not others.")
    ```
*   **Display (if results available and plots generated):**
    ```python
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
    ```

#### Page: "7. Alternative Data Evaluation Framework"

*   **Markdown:**
    ```python
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
    ```
*   **Widgets:**
    *   `st.button("Evaluate Alt Data Scorecard")` triggers evaluation.
*   **Pre-conditions**: `evaluation` and `weights` dictionaries are available from `source.py` (due to `from source import *`).
*   **Function Calls (on button click):**
    *   `st.session_state.overall_alt_data_score = evaluate_alt_data_scorecard(evaluation, weights)`
*   **Session State Update**: `st.session_state.overall_alt_data_score`.
*   **Markdown:**
    ```python
    st.markdown(f"The scorecard for Google Trends reveals its strengths and weaknesses. While it scores highly on `History`, `Legality`, and `Cost` (being free and publicly available), it scores only moderately on `Predictive Power` and poorly on `Uniqueness` and `Coverage`. This highlights the \"alt data paradox\": the best data is often expensive and proprietary, while free data is usually crowded. For *Alpha Insights*, this means Google Trends data is a good starting point for exploratory research and may contribute to a multi-signal model, but it's unlikely to be a standalone alpha source due to its widespread use and limited uniqueness. This structured assessment is vital for communicating the true value and limitations of alternative data to our investment committee.")
    ```
*   **Display (if score calculated):**
    ```python
    if st.session_state.overall_alt_data_score is not None:
        st.markdown("### Alternative Data Evaluation Scorecard for Google Trends")
        scorecard_df = pd.DataFrame.from_dict(evaluation, orient='index', columns=['Score'])
        scorecard_df.index.name = 'Dimension'
        scorecard_df = scorecard_df.drop('Dataset')
        st.dataframe(scorecard_df)
        st.markdown(f"**Overall Alt Data Score: {st.session_state.overall_alt_data_score:.1f} / 5.0**")
    ```
