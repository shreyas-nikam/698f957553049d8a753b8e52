# QuLab: Lab 9: Alternative Data Signals - Google Trends for Stock Prediction

![Streamlit Badge](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

## Project Title: Alternative Data Signals: Google Trends for Retail Revenue & Stock Performance Prediction

## Project Description

This Streamlit application, part of the QuLab series (Lab 9), serves as a practical lab project designed to demonstrate the end-to-end workflow for evaluating alternative data, specifically Google Trends search interest, as a predictive signal for stock performance.

The project is framed around a **persona**: Sarah Chen, CFA, an Equity Analyst at Alpha Insights, a quantitative asset management firm.

**Scenario**: Sarah is tasked with investigating whether surges in Google search interest for a leading consumer brand (e.g., Nike, Apple, Tesla, Google) can provide an early indicator of consumer demand shifts and, consequently, anticipate earnings surprises or movements in stock prices. The application guides users through a systematic approach, from raw data acquisition and statistical validation to the construction of a prototype signal and a structured evaluation of the alternative data source itself.

**Key Learnings & Focus Areas:**
*   **Alternative Data Sourcing**: Understanding the nuances of fetching and handling public alternative data (Google Trends).
*   **Data Preprocessing**: Aligning, normalizing, and engineering features from disparate data sources.
*   **Statistical Validation**: Applying econometric techniques like cross-correlation and Granger Causality to establish predictive relationships.
*   **Signal Prototyping**: Constructing a simple trading signal and evaluating its performance using industry-standard metrics (Sharpe Ratio, IC, ICIR).
*   **Alternative Data Evaluation Framework**: Assessing data quality, legality, cost, and uniqueness beyond purely quantitative performance.

**Note on Dummy Data**: For self-contained execution and ease of demonstration in a lab environment, this application uses **dummy implementations** for data fetching (Google Trends, Financial Data), preprocessing, and statistical functions. In a real-world scenario, these would connect to live APIs (e.g., `pytrends`, `yfinance`) and implement more complex statistical models (e.g., `statsmodels` for Granger Causality). The dummy data simulates plausible trends and correlations to illustrate the analytical workflow.

## Features

The application is structured into several interactive sections, accessible via the sidebar navigation:

1.  **Introduction**: Sets the context with the persona and scenario.
2.  **Data Acquisition**:
    *   Allows selection of a company ticker and related search term.
    *   Specifies a date range for data fetching.
    *   **Dummy Functionality**: Simulates fetching Google Trends search interest and historical stock prices.
    *   Displays sample raw dataframes.
3.  **Data Preprocessing & Feature Engineering**:
    *   User-definable moving average window.
    *   **Dummy Functionality**: Aligns, resamples, and merges trends and financial data.
    *   Calculates stock returns.
    *   Engineers features: Z-scored search volume, moving average of search volume, and search volume deviation from MA.
    *   Displays sample processed data and descriptive statistics.
4.  **Exploratory Visual Analysis**:
    *   Generates a dual-axis time-series plot comparing Z-scored search volume against cumulative stock returns, aiding in visual pattern identification.
5.  **Lead-Lag Cross-Correlation Analysis**:
    *   User-definable maximum lag for analysis.
    *   **Dummy Functionality**: Calculates and plots the cross-correlation between search volume deviation and weekly stock returns for various lags.
    *   Includes significance bounds to assess statistical relevance.
6.  **Granger Causality Testing**:
    *   User-definable maximum lag for the test.
    *   **Dummy Functionality**: Performs a simulated Granger Causality test to determine if past search volume statistically predicts future returns.
    *   Outputs simulated F-statistics and p-values with conclusions.
7.  **Prototype Signal Construction & Performance Evaluation**:
    *   **Dummy Functionality**: Constructs a simple prototype signal (e.g., buy if search volume > MA).
    *   Evaluates signal performance against a buy-and-hold benchmark using metrics like Sharpe Ratio, Cumulative Return, Hit Rate, Information Coefficient (IC), and IC Information Ratio (ICIR).
    *   Generates plots for cumulative returns comparison, rolling IC decay, and a scatter plot of search deviation vs. next-week returns.
8.  **Alternative Data Evaluation Framework**:
    *   **Dummy Functionality**: Provides a structured scorecard to evaluate the alternative data source (Google Trends) across qualitative dimensions like Predictive Power, Uniqueness, Coverage, Timeliness, History, Legality, and Cost.
    *   Calculates an overall weighted score.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository** (or download the `app.py` file if no repository is provided):
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name # Replace with your actual repo name
    ```
    If you only have the `app.py` file, simply navigate to its directory.

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required packages**:
    Create a `requirements.txt` file in the same directory as `app.py` with the following content:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application**:
    Ensure your virtual environment is activated and you are in the directory containing `app.py`.
    ```bash
    streamlit run app.py
    ```

2.  **Interact with the Application**:
    *   Your web browser should automatically open to `http://localhost:8501`.
    *   Use the **sidebar navigation** to move through the different sections of the lab project.
    *   Adjust parameters (e.g., dates, MA window, lags) in each section and click the respective buttons to trigger data processing, analysis, and plot generation.
    *   Review the displayed dataframes, plots, and text outputs.

## Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── requirements.txt        # Python dependencies
└── README.md               # This README file
├── trends_vs_returns.png   # (Generated at runtime) Exploratory plot
├── lead_lag_cross_correlation.png # (Generated at runtime) Cross-correlation plot
├── cumulative_returns_comparison.png # (Generated at runtime) Signal performance plot
├── rolling_ic_decay.png    # (Generated at runtime) Signal performance plot
└── search_dev_vs_next_return_scatter.png # (Generated at runtime) Signal performance plot
```
*Note: In a more complex, production-ready application, data fetching, processing, and plotting functions would typically reside in separate utility modules (e.g., `src/data.py`, `src/processing.py`, `src/plotting.py`) and be imported into `app.py`.*

## Technology Stack

*   **Application Framework**: [Streamlit](https://streamlit.io/)
*   **Programming Language**: Python
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Data Visualization**: [Matplotlib](https://matplotlib.org/)
*   **Utility**: `io`, `sys`, `os` (standard Python libraries)

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (If you don't have a LICENSE file, you might want to create one, or just state "No specific license, for educational purposes.")

## Contact

For any questions or feedback regarding this project, please reach out:

*   **Quant University** - [https://www.quantuniversity.com/](https://www.quantuniversity.com/)
*   **Project Maintainer**: [Your Name/Organization Name] - [your_email@example.com](mailto:your_email@example.com) (Optional)
