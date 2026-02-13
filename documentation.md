id: 698f957553049d8a753b8e52_documentation
summary: Lab 9: Alternative Data Signals Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Lab 9: Building and Evaluating Alternative Data Signals with Streamlit

## 1. Introduction to Alternative Data & Application Overview
Duration: 00:05

Welcome to QuLab: Lab 9, where we delve into the exciting world of **Alternative Data Signals** using a powerful Streamlit application. This codelab is designed to provide a comprehensive guide for developers and quantitative analysts to understand, interact with, and extend the functionalities of this Streamlit app.

**The Importance of Alternative Data in Finance:**
In today's competitive financial markets, traditional financial data (e.g., stock prices, earnings reports) often reflects information already priced into securities. Alternative data, which encompasses non-traditional datasets like satellite imagery, social media sentiment, web traffic, and in our case, Google Trends search interest, offers a potential edge. It aims to capture real-world economic activity or shifts in consumer behavior *before* they are fully reflected in traditional financial metrics, thus providing opportunities to generate alpha.

**Our Persona and Scenario:**
Imagine you are **Sarah Chen, CFA, an Equity Analyst at Alpha Insights**, a quantitative asset management firm. Your mission is to find new sources of alpha. You are investigating whether public search interest for a leading consumer brand, like **Nike ($NKE)**, can serve as an early indicator of consumer demand shifts and anticipate stock price movements or earnings surprises. This application guides you through a systematic workflow to evaluate such an alternative data source.

**Application Overview:**
This Streamlit application provides an end-to-end workflow for analyzing alternative data, specifically Google Trends, in conjunction with traditional financial data. It allows users to:
1.  **Acquire Data:** Fetch Google Trends search interest and historical stock prices.
2.  **Preprocess & Engineer Features:** Clean, align, normalize data, and create predictive features.
3.  **Explore Visually:** Identify initial patterns between search interest and stock performance.
4.  **Quantify Relationships:** Use cross-correlation to measure lead-lag relationships.
5.  **Test Predictive Power:** Apply Granger Causality tests for statistical evidence of forecasting ability.
6.  **Build & Evaluate Signals:** Construct a prototype signal and assess its performance.
7.  **Framework Evaluation:** Use a structured scorecard to holistically evaluate the alternative data source.

**Application Architecture and Data Flow:**

The Streamlit application acts as an interactive user interface (UI) that orchestrates a series of data processing and analysis steps. Here's a high-level conceptual flow:

<pre>
++     +--+     +--+
| User Inputs (Streamlit) | --> | 1. Data Acquisition (Dummy)  | --> | Raw Google Trends Data   |
|     - Ticker        |     |   (get_google_trends_data) |     | Raw Financial Data       |
|     - Search Term   |     |   (get_financial_data)     |     +--+
|     - Dates         |     +--+                  |
++                                                    V
                                                          +-+
                                                          | 2. Data Preprocessing & Feature     |
                                                          |    Engineering (preprocess_data)    |
                                                          |    - Resampling, Alignment          |
                                                          |    - Z-scoring, MA Deviation        |
                                                          +-+
                                                                           |
                                                                           V
                                                          +-+
                                                          | Processed Data with Engineered      |
                                                          | Features                            |
                                                          +-+
                                                                           |
                  +S--## 1. Introduction to Alternative Data & Application Overview
Duration: 00:05

Welcome to **QuLab: Lab 9**, where we delve into the exciting world of **Alternative Data Signals** using a powerful Streamlit application. This codelab is designed to provide a comprehensive guide for developers and quantitative analysts to understand, interact with, and extend the functionalities of this Streamlit app.

**The Importance of Alternative Data in Finance:**
In today's competitive financial markets, traditional financial data (e.g., stock prices, earnings reports) often reflects information already priced into securities. Alternative data, which encompasses non-traditional datasets like satellite imagery, social media sentiment, web traffic, and in our case, Google Trends search interest, offers a potential edge. It aims to capture real-world economic activity or shifts in consumer behavior *before* they are fully reflected in traditional financial metrics, thus providing opportunities to generate alpha.

**Our Persona and Scenario:**
Imagine you are **Sarah Chen, CFA, an Equity Analyst at Alpha Insights**, a quantitative asset management firm. Your mission is to find new sources of alpha. You are investigating whether public search interest for a leading consumer brand, like **Nike ($NKE)**, can serve as an early indicator of consumer demand shifts and anticipate stock price movements or earnings surprises. This application guides you through a systematic workflow to evaluate such an alternative data source.

**Application Overview:**
This Streamlit application provides an end-to-end workflow for analyzing alternative data, specifically Google Trends, in conjunction with traditional financial data. It allows users to:
1.  **Acquire Data:** Fetch Google Trends search interest and historical stock prices.
2.  **Preprocess & Engineer Features:** Clean, align, normalize data, and create predictive features.
3.  **Explore Visually:** Identify initial patterns between search interest and stock performance.
4.  **Quantify Relationships:** Use cross-correlation to measure lead-lag relationships.
5.  **Test Predictive Power:** Apply Granger Causality tests for statistical evidence of forecasting ability.
6.  **Build & Evaluate Signals:** Construct a prototype signal and assess its performance.
7.  **Framework Evaluation:** Use a structured scorecard to holistically evaluate the alternative data source.

**Application Architecture and Data Flow:**

The Streamlit application acts as an interactive user interface (UI) that orchestrates a series of data processing and analysis steps. Here's a high-level conceptual flow:

```
++     +--+     +--+
| User Inputs (Streamlit) | --> | 1. Data Acquisition (Dummy)  | --> | Raw Google Trends Data   |
|     - Ticker        |     |   (get_google_trends_data) |     | Raw Financial Data       |
|     - Search Term   |     |   (get_financial_data)     |     +--+
|     - Dates         |     +--+                  |
++                                                    V
                                                          +-+
                                                          | 2. Data Preprocessing & Feature     |
                                                          |    Engineering (preprocess_data)    |
                                                          |    - Resampling, Alignment          |
                                                          |    - Z-scoring, MA Deviation        |
                                                          +-+
                                                                           |
                                                                           V
                                                          +-+
                                                          | Processed Data with Engineered      |
                                                          | Features                            |
                                                          +-+
                                                                           |
                  +-1.  **Fenced Code and Language Hints:** Code blocks (e-g, Python code)
```python
import streamlit as st
```

This ensures proper syntax highlighting for readability.

#### Info Boxes

Info boxes are colored callouts that enclose special information in codelabs. Positive info boxes should contain positive information like best practices and time-saving tips. Negative infoboxes should contain information like warnings and API usage restriction. If you want to highlight important information, use the <b> tag inside the aside tag.

```html
<aside class="positive">
This will appear in a <b>positive</b> info box.
</aside>

<aside class="negative">
This will appear in a <b>negative</b> info box.
</aside>
```

#### Download Buttons

Codelabs sometimes contain links to SDKs or sample code. The codelab renderer will apply special button-esque styling to any link that begins with the word "Download".

```html
<button>
  [Download SDK](https://www.google.com)
</button>
```



## 1. Introduction to Alternative Data & Application Overview
Duration: 00:05

Welcome to **QuLab: Lab 9**, where we delve into the exciting world of **Alternative Data Signals** using a powerful Streamlit application. This codelab is designed to provide a comprehensive guide for developers and quantitative analysts to understand, interact with, and extend the functionalities of this Streamlit app.

**The Importance of Alternative Data in Finance:**
In today's competitive financial markets, traditional financial data (e.g., stock prices, earnings reports) often reflects information already priced into securities. Alternative data, which encompasses non-traditional datasets like satellite imagery, social media sentiment, web traffic, and in our case, Google Trends search interest, offers a potential edge. It aims to capture real-world economic activity or shifts in consumer behavior *before* they are fully reflected in traditional financial metrics, thus providing opportunities to generate alpha.

**Our Persona and Scenario:**
Imagine you are **Sarah Chen, CFA, an Equity Analyst at Alpha Insights**, a quantitative asset management firm. Your mission is to find new sources of alpha. You are investigating whether public search interest for a leading consumer brand, like **Nike ($NKE)**, can serve as an early indicator of consumer demand shifts and anticipate stock price movements or earnings surprises. This application guides you through a systematic workflow to evaluate such an alternative data source.

**Application Overview:**
This Streamlit application provides an end-to-end workflow for analyzing alternative data, specifically Google Trends, in conjunction with traditional financial data. It allows users to:
1.  **Acquire Data:** Fetch Google Trends search interest and historical stock prices.
2.  **Preprocess & Engineer Features:** Clean, align, normalize data, and create predictive features.
3.  **Explore Visually:** Identify initial patterns between search interest and stock performance.
4.  **Quantify Relationships:** Use cross-correlation to measure lead-lag relationships.
5.  **Test Predictive Power:** Apply Granger Causality tests for statistical evidence of forecasting ability.
6.  **Build & Evaluate Signals:** Construct a prototype signal and assess its performance.
7.  **Framework Evaluation:** Use a structured scorecard to holistically evaluate the alternative data source.

**Application Architecture and Data Flow:**

The Streamlit application acts as an interactive user interface (UI) that orchestrates a series of data processing and analysis steps. Here's a high-level conceptual flow:

