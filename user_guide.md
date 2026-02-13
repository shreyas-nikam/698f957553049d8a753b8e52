id: 698f957553049d8a753b8e52_user_guide
summary: Lab 9: Alternative Data Signals User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Lab 9: Alternative Data Signals for Stock Prediction

## 1. Introduction: Unlocking Market Insights with Alternative Data
Duration: 0:05
Welcome to QuLab: Lab 9, where we explore the exciting world of **Alternative Data Signals** for financial market prediction.

<aside class="positive">
<b>The Goal:</b> This codelab will guide you through using Google Trends as an alternative data source to potentially predict stock performance for consumer-facing companies. You'll learn the workflow from data acquisition to signal evaluation, mimicking the process of a quantitative equity analyst.
</aside>

**Persona: Sarah Chen, CFA, Equity Analyst at Alpha Insights**

**Scenario:**
"As an equity analyst at *Alpha Insights*, a quantitative asset management firm, I'm constantly seeking new informational edges to generate alpha for our portfolios. Traditional financial data, while essential, often reflects information already priced into the market. My current focus is on alternative data—specifically, public search interest captured by Google Trends—to see if it can provide an early indicator of consumer demand shifts and, consequently, anticipate earnings surprises or movements in stock prices for consumer-facing companies."

"Today, I'm examining a leading consumer brand, **Nike ($NKE)**, to investigate whether surges in search interest for their products precede increases in sales or stock returns. This real-world workflow will take me from raw data acquisition through statistical validation to the construction of a prototype signal, and finally, a structured evaluation of the alternative data source itself. This systematic approach is critical for incorporating non-traditional data responsibly into our investment strategies."

Use the sidebar on the left to navigate through the workflow steps. Start by selecting "Introduction" if you haven't already, then move to "1. Data Acquisition".

## 2. Data Acquisition: Gathering the Raw Materials
Duration: 0:05
As an analyst at *Alpha Insights*, my first step is always to gather the necessary data. For this investigation, I need two primary sources: historical Google Trends search interest for brand-related terms and the brand's historical stock prices. The challenge with alternative data often begins with sourcing—ensuring I retrieve comprehensive and relevant data while adhering to API best practices.

**Your Task:**
1.  In the Streamlit sidebar, ensure "1. Data Acquisition" is selected.
2.  Use the dropdowns to **"Select a Company Ticker"** (e.g., `NKE` for Nike).
3.  Choose a **"Search Term"** relevant to the selected company (e.g., `Nike` for NKE).
4.  Adjust the **"Start Date"** and **"End Date"** as desired. The default range is usually a good starting point.
5.  Click the **"Fetch Data"** button.

Once clicked, the application will simulate fetching the relevant Google Trends data for your selected search term and financial data for the chosen ticker.

The raw Google Trends data provides search interest as a relative index, not absolute volume. This means the values (0-100) are normalized to the peak interest within the queried time window, and they can vary slightly with repeated API calls due to sampling. This is a common characteristic of free alternative data—it requires careful preprocessing. The financial data, on the other hand, provides clean historical adjusted close prices.

<aside class="negative">
<b>Practitioner Warning:</b> Google Trends data is noisy and non-reproducible. Repeated API calls for the same query can return slightly different values because Google samples from its full search database. The index is also relative to the query window—changing the start/end date changes all values. Best practices: (a) always retrieve the full desired window in a single call, (b) average multiple calls to reduce sampling noise, (c) document the exact retrieval date and parameters. For production use, institutional investors subscribe to vendors (e.g., Quandl/Nasdaq Data Link) that provide cleaned, versioned Google Trends data. For this teaching exercise, the raw pytrends output is sufficient.
</aside>

You will see sample dataframes of the fetched Google Trends and Financial data displayed below the fetch button.

## 3. Data Preprocessing & Feature Engineering: Preparing for Analysis
Duration: 0:07
Raw data, especially alternative data, is rarely ready for direct analysis. As an equity analyst, I need to perform critical preprocessing steps to ensure data quality, consistency, and to extract meaningful features. This involves:

1.  **Resampling and Alignment:** Both Google Trends and stock price data might have slightly different timestamps or frequencies. I need to align them to a consistent weekly frequency (e.g., week-ending dates) to enable direct comparison.
2.  **Normalization:** The Google Trends index is relative. To compare search interest meaningfully across different companies or even different time periods for the same company, I'll normalize it using z-scores. This scales the data to have a mean of 0 and a standard deviation of 1.
3.  **Feature Engineering:** Raw search volume might not be the most predictive signal. Changes in search volume (Week-over-Week or Year-over-Year) or deviations from its moving average might better capture shifts in consumer sentiment or momentum. I'll compute these as potential predictive features.

The z-score normalization for Google Trends index $G_{i,t}$ for company $i$ at week $t$ is given by:
$$Z_{i,t} = \frac{G_{i,t} - \bar{G_i}}{\sigma_{G_i}}$$
where $\bar{G_i}$ is the mean of $G_i$ and $\sigma_{G_i}$ is the standard deviation of $G_i$ over the sample period.

I will also calculate the moving average deviation ($D_{i,t}$) which measures how far the current search interest is from its recent trend, using a 12-week moving average ($MA_{12w}$):
$$D_{i,t} = G_{i,t} - MA_{12w}(G_i)$$

**Your Task:**
1.  In the Streamlit sidebar, select "2. Data Preprocessing & Feature Engineering".
2.  If you haven't fetched data in the previous step, you'll see an error message. Please go back to Step 2 to fetch data.
3.  Adjust the **"Moving Average Window (weeks)"** if desired (default is 12). This window determines the length of the historical period used to calculate the moving average.
4.  Click the **"Process Data"** button.

Once processed, the application will display sample data from the merged and feature-engineered dataset, including new columns like `search_z` (z-scored search volume) and `search_ma_dev` (moving average deviation of search volume). Z-scoring allows me to compare search interest trends across different companies, even if their absolute search volumes vary greatly. The moving average deviation, in particular, helps identify when search interest is unusually high or low relative to its recent trend. This kind of deviation could signal a shift in consumer attention that might precede changes in financial performance, making it a potentially strong predictive feature for my models at *Alpha Insights*.

## 4. Exploratory Visual Analysis: Spotting Initial Patterns
Duration: 0:05
Before diving into complex statistical models, I always start with visual exploration. As an equity analyst, I'm looking for intuitive lead-lag patterns—do spikes in '{st.session_state.selected_ticker}' search interest visually precede upticks in its stock performance? This helps me form initial hypotheses and sanity-check the data before more rigorous analysis. I'll create dual-axis time-series charts, plotting the normalized Google Trends index against the cumulative stock return for each company.

**Your Task:**
1.  In the Streamlit sidebar, select "3. Exploratory Visual Analysis".
2.  If you haven't processed data in the previous step, you'll see an error message. Please go back to Step 3 to process data.
3.  Click the **"Generate Dual-Axis Time-Series Chart"** button.

You'll see a chart plotting the Z-scored Search Volume (blue line) against the Cumulative Returns (red line) for your selected ticker. For consumer-facing companies like Nike, I often observe that strong surges in search interest, especially when normalized, visually precede periods of positive stock performance. This informal observation provides initial support for my hypothesis, but it requires rigorous statistical validation. These charts are invaluable for quickly grasping potential relationships and identifying periods where the correlation might be strongest or weakest.

## 5. Lead-Lag Cross-Correlation Analysis: Quantifying Relationships
Duration: 0:07
Visual inspection is a good starting point, but as an analyst at *Alpha Insights*, I need to quantify these lead-lag relationships formally. The cross-correlation function (CCF) measures the similarity between two time series, '{st.session_state.selected_ticker}' search volume ($x$) and '{st.session_state.selected_ticker}' stock returns ($y$), as a function of the lag applied to one of them. Specifically, I want to see if search volume at time $t$ predicts stock returns at a future time $t+k$.

The cross-correlation at lag $k$, denoted $P_{xy}(k)$, is calculated as:
$$P_{xy}(k) = \frac{\sum_{t=1}^{T-k} (x_t - \bar{x}) (y_{t+k} - \bar{y})}{\sqrt{\sum_{t=1}^{T} (x_t - \bar{x})^2 \sum_{t=1}^{T} (y_t - \bar{y})^2}}$$
where $x_t$ is the search volume (or its deviation) at time $t$, $y_t$ is the stock return at time $t$, $\bar{x}$ and $\bar{y}$ are their respective means, and $T$ is the number of observations.

*   If $k > 0$, it means $x$ (search volume) leads $y$ (returns) by $k$ periods. I'm particularly interested in positive lags, as they suggest predictive power.
*   If $k < 0$, it means $x$ lags $y$ by $|k|$ periods.
*   If $k = 0$, it's the simultaneous correlation.

To determine if a correlation is statistically meaningful, I compare it against significance bounds. Under the null hypothesis of no correlation, the approximate 95% confidence interval is $\pm 1.96 / \sqrt{T}$, where $T$ is the number of observations. Correlations outside these bounds are statistically significant at the 5% level, suggesting a true relationship rather than random noise. For weekly data over 5 years (approx. $T=260$ weeks), this bound is roughly $\pm 1.96 / \sqrt{260} \approx \pm 0.12$.

**Your Task:**
1.  In the Streamlit sidebar, select "4. Lead-Lag Cross-Correlation Analysis".
2.  If you haven't processed data in the previous step, you'll see an error message. Please go back to Step 3 to process data.
3.  Adjust the **"Max Lag for Cross-Correlation (weeks)"** as desired (default is 8).
4.  Click the **"Calculate & Plot Cross-Correlations"** button.

The bar chart will reveal the cross-correlation values at different lags. For companies like Nike, search volume often exhibits positive correlation with future stock returns at lags of 1-4 weeks, and these correlations frequently exceed the calculated significance bounds. This suggests that public interest in the brand, as captured by Google Trends, could indeed be an early indicator of market movement, reinforcing my initial hypothesis. For Apple, the correlations might be weaker due to its larger market capitalization and faster information dissemination. This quantitative insight is a critical step in building a data-driven investment thesis.

## 6. Granger Causality Testing: Statistical Predictive Power
Duration: 0:07
Correlation indicates a relationship, but it does not imply that one variable *causes* or *predicts* another in a statistical sense. For *Alpha Insights*, I need stronger evidence of predictive power. The Granger causality test formalizes whether past values of '{st.session_state.selected_ticker}' search volume ($x$) statistically improve forecasts of future stock returns ($y$), beyond what past returns alone provide. This is crucial for evaluating a signal's potential robustness.

The test compares two models:
1.  **Restricted Model ($H_0$):** This model assumes that past values of $x$ (search volume) do not help predict $y$ (returns). It only uses past values of $y$:
    $$y_t = \alpha + \sum_{j=1}^{p} \phi_j y_{t-j} + \epsilon_t$$
2.  **Unrestricted Model ($H_1$):** This model assumes that past values of $x$ *do* help predict $y$. It includes both past values of $y$ and past values of $x$:
    $$y_t = \alpha + \sum_{j=1}^{p} \phi_j y_{t-j} + \sum_{j=1}^{p} \gamma_j x_{t-j} + \eta_t$$
The F-statistic is then computed to compare the sum of squared residuals (SSR) from these two models:
$$F = \frac{(SSR_{restricted} - SSR_{unrestricted})/p}{SSR_{unrestricted}/(T - 2p - 1)} \sim F(p, T - 2p - 1)$$
where $p$ is the number of lags, and $T$ is the number of observations.

I reject the null hypothesis ($H_0$: search volume does NOT Granger-cause returns) if the F-test p-value is below a chosen significance level (e.g., 0.05). This provides statistical evidence that past search volume has predictive power for future returns. However, it's an *important caveat* that Granger causality in financial time-series can be fragile; a relationship observed in-sample might not hold out-of-sample due to market adaptation or other factors.

**Your Task:**
1.  In the Streamlit sidebar, select "5. Granger Causality Testing".
2.  If you haven't processed data in the previous step, you'll see an error message. Please go back to Step 3 to process data.
3.  Adjust the **"Max Lag for Granger Causality Test (weeks)"** (default is 4).
4.  Click the **"Perform Granger Causality Test"** button.

The output will show the results of the Granger Causality test. For Nike, the Granger causality test often yields p-values below 0.05 for lags of 1-2 weeks when examining `search_ma_dev`'s influence on returns. This suggests that past deviations in search volume indeed help predict future stock returns, providing statistical backing to my hypothesis. For companies like Apple, which are highly efficient and incorporate information quickly, the p-values might be higher, indicating a weaker or non-existent predictive relationship from this specific alternative data source. This statistical validation is crucial for deciding whether to integrate such a signal into our quantitative models at *Alpha Insights*.

## 7. Prototype Signal Construction & Performance Evaluation: Building and Testing an Edge
Duration: 0:10
With statistical evidence of predictive power, I can now construct a simple prototype investment signal for *Alpha Insights*. My rule will be straightforward: if '{st.session_state.selected_ticker}' search volume (`search_volume`) is above its 12-week moving average (`search_ma`), I generate a 'buy' signal (signal = 1); otherwise, the signal is 0 (neutral/bearish). I'll then simulate its performance and evaluate it against a simple buy-and-hold benchmark using a suite of metrics.

Key evaluation metrics for alternative data signals include:
*   **Sharpe Ratio:** Measures risk-adjusted return.
*   **Cumulative Return:** Total return over the period.
*   **Hit Rate:** Percentage of weeks where the signal correctly predicts the direction of returns.
*   **Information Coefficient (IC):** Measures the rank correlation between the signal and the subsequent outcome.
    $$IC_t = Spearman(S_{i,t}, r_{i,t+1})$$
    where $S_{i,t}$ is the Google Trends signal for stock $i$ at time $t$ and $r_{i,t+1}$ is the forward return (e.g., next-week return). The average IC across $T$ time periods is:
    $$\overline{IC} = \frac{1}{T}\sum_{t=1}^{T} IC_t$$
    A meaningful signal typically has an $|\overline{IC}| > 0.05$.
*   **IC Information Ratio (ICIR):** Assesses the consistency of the signal, analogous to the Sharpe Ratio of the IC.
    $$ICIR = \frac{\overline{IC}}{\sigma_{IC}}$$
    where $\sigma_{IC}$ is the standard deviation of the ICs. An $ICIR > 0.5$ indicates a consistent and potentially robust signal.
*   **Signal Decay Analysis:** Plotting the rolling IC over time (e.g., 2-year windows) helps detect if the signal's predictive power is weakening, possibly due to market adaptation or crowding of the signal.

**Your Task:**
1.  In the Streamlit sidebar, select "6. Prototype Signal Construction & Performance Evaluation".
2.  If you haven't processed data in the previous step, you'll see an error message. Please go back to Step 3 to process data.
3.  Click the **"Evaluate Signal Performance"** button.

The application will generate three plots:
*   **Cumulative Returns Comparison:** Shows the growth of $1 invested in the signal vs. a buy-and-hold benchmark.
*   **Rolling IC (Signal Decay):** Plots the Information Coefficient over rolling windows to observe its consistency and potential decay.
*   **Search Change vs. Forward Return Scatter Plot:** Visualizes the relationship between our search deviation feature and next week's stock returns.

It will also display a **Signal Performance Summary** table with metrics like Sharpe Ratio, Cumulative Return, Hit Rate, Average IC, and ICIR.

For Nike, the prototype signal shows a modest improvement in Sharpe Ratio and a positive hit rate, indicating it correctly predicts positive returns more than 50% of the time. The average IC is positive, suggesting some predictive power, although it might fall into the 'meaningful' rather than 'strong' category. The rolling IC plot helps me identify if the signal's effectiveness is decaying over time, which is common with crowded alternative data sources. The scatter plot further confirms a slight positive relationship between search moving average deviation and next-week stock returns.

<aside class="negative">
<b>Practitioner Warning:</b> This is a prototype, not a production strategy. A viable alternative data signal for *Alpha Insights* would require multi-year out-of-sample testing, rigorous transaction cost modeling, and, most importantly, combination with other signals in a multi-factor model. Free alternative data often produces marginal signals due to market efficiency and crowding risk. Realistic expectations for Google Trends-based signals include an IC around 0.02-0.05 and a Sharpe improvement of 0.1-0.3 versus buy-and-hold, and results that are statistically significant for some companies but not others.
</aside>

## 8. Alternative Data Evaluation Framework: Beyond the Numbers
Duration: 0:05
As a CFA charterholder at *Alpha Insights*, my responsibility extends beyond quantitative performance metrics. I must also conduct a structured evaluation of the alternative dataset itself. This framework helps me assess Google Trends—or any other alternative data source—across key dimensions that cover its potential value, limitations, and risks. This ensures we make informed decisions about integrating new data into our investment process, considering both quantitative results and qualitative factors like legal implications and crowding risk.

The dimensions I consider are:
*   **Predictive Power:** Does the data contain information about future prices or fundamentals? (Quantified by IC, Granger causality p-value).
*   **Uniqueness:** Is this signal differentiated, or does everyone have it? (Crowding risk).
*   **Coverage:** How many securities does it cover? Is it biased?
*   **Timeliness:** How quickly is the data available?
*   **History:** How many years of backtest data exist?
*   **Legality:** Is the data legally obtained? Any Material Nonpublic Information (MNPI) concerns? Web scraping restrictions?
*   **Cost:** Free? Subscription? One-time?

I will assign scores (e.g., 1-5 scale) and weights to each dimension to calculate an overall score, providing a holistic view of Google Trends as an alternative data source.

**Your Task:**
1.  In the Streamlit sidebar, select "7. Alternative Data Evaluation Framework".
2.  Click the **"Evaluate Alt Data Scorecard"** button.

The scorecard for Google Trends will be displayed, along with an overall score. The scorecard for Google Trends reveals its strengths and weaknesses. While it scores highly on `History`, `Legality`, and `Cost` (being free and publicly available), it scores only moderately on `Predictive Power` and poorly on `Uniqueness` and `Coverage`. This highlights the "alt data paradox": the best data is often expensive and proprietary, while free data is usually crowded. For *Alpha Insights*, this means Google Trends data is a good starting point for exploratory research and may contribute to a multi-signal model, but it's unlikely to be a standalone alpha source due to its widespread use and limited uniqueness. This structured assessment is vital for communicating the true value and limitations of alternative data to our investment committee.

Congratulations! You have completed the QuLab: Lab 9, demonstrating a comprehensive workflow for evaluating alternative data signals using Google Trends.
