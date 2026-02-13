
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
import os
import sys

# Mock the 'source' module and its contents
mock_queries = {
    'NKE': ['Nike Shoes', 'Nike Stock'],
    'AAPL': ['Apple iPhone', 'Apple Stock']
}

mock_trends_raw_data = {
    'NKE': pd.DataFrame({
        'Nike Shoes': [50, 55, 60, 65, 70],
        'isPartial': [False, False, False, False, False]
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=5, freq='W')))
}

mock_financial_raw_data = pd.DataFrame({
    'NKE': [100, 102, 101, 105, 103]
}, index=pd.to_datetime(pd.date_range('2023-01-01', periods=5, freq='W')))

mock_processed_data = {
    'NKE': pd.DataFrame({
        'search_z': np.random.rand(5),
        'search_ma_dev': np.random.rand(5),
        'returns': np.random.rand(5),
        'search_ma': np.random.rand(5),
        'signal': [0, 1, 0, 1, 0]
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=5, freq='W')))
}

mock_signal_performance_summary = pd.DataFrame({
    'Metric': ['Sharpe Ratio', 'Cumulative Return', 'Hit Rate', 'Average IC', 'ICIR'],
    'Value': [1.2, 0.15, 0.6, 0.03, 0.7]
})

# Mock for evaluation and weights for the scorecard
mock_evaluation = {
    'Dataset': 'Google Trends',
    'Predictive Power': 3,
    'Uniqueness': 2,
    'Coverage': 3,
    'Timeliness': 4,
    'History': 5,
    'Legality': 5,
    'Cost': 5
}
mock_weights = {
    'Predictive Power': 0.3,
    'Uniqueness': 0.2,
    'Coverage': 0.15,
    'Timeliness': 0.1,
    'History': 0.05,
    'Legality': 0.1,
    'Cost': 0.1
}

# Patching the source module's contents at the module level for all tests
with patch('source.queries', new=mock_queries), \
     patch('source.get_google_trends_data', return_value=mock_trends_raw_data), \
     patch('source.get_financial_data', return_value=mock_financial_raw_data), \
     patch('source.preprocess_data', return_value=mock_processed_data), \
     patch('source.plot_dual_axis_trends_vs_returns', return_value=None), \
     patch('source.plot_cross_correlations', return_value=None), \
     patch('source.perform_granger_causality', return_value=None), \
     patch('source.evaluate_signal_performance', return_value=mock_signal_performance_summary), \
     patch('source.evaluate_alt_data_scorecard', return_value=3.5), \
     patch('source.evaluation', new=mock_evaluation, create=True), \
     patch('source.weights', new=mock_weights, create=True), \
     patch('os.path.exists', return_value=True): # Mock os.path.exists for image checks

    def get_app_test_instance(page_name="Introduction"):
        """Helper to get an AppTest instance and set the page."""
        at = AppTest.from_file("app.py")
        at.session_state["selected_ticker"] = "NKE" # Ensure a default ticker is set
        at.session_state["search_term"] = mock_queries['NKE'][0] # Ensure a default search term
        at.run() # Initial run to populate sidebar selectbox options
        at.sidebar.selectbox[0].set_value(page_name).run()
        return at

    def test_introduction_page():
        at = get_app_test_instance("Introduction")
        assert at.markdown[0].value.startswith("## Alternative Data Signals:")
        assert "Sarah Chen, CFA, Equity Analyst at Alpha Insights" in at.markdown[1].value

    def test_data_acquisition_page_initial_load():
        at = get_app_test_instance("1. Data Acquisition")
        assert at.selectbox[0].value == "NKE"
        assert at.selectbox[1].value == "Nike Shoes"
        assert at.date_input[0].value == pd.to_datetime('2019-01-01').date()
        assert at.date_input[1].value == pd.to_datetime('2024-01-01').date()

    def test_data_acquisition_page_fetch_data():
        at = get_app_test_instance("1. Data Acquisition")

        # Simulate user interaction
        at.selectbox[0].set_value("AAPL").run()
        at.selectbox[1].set_value("Apple iPhone").run()
        at.date_input[0].set_value(pd.to_datetime('2020-01-01').date()).run()
        at.date_input[1].set_value(pd.to_datetime('2023-01-01').date()).run()
        at.button[0].click().run()

        # Verify session state updates
        assert at.session_state.selected_ticker == "AAPL"
        assert at.session_state.search_term == "Apple iPhone"
        assert at.session_state.start_date == pd.to_datetime('2020-01-01').date()
        assert at.session_state.end_date == pd.to_datetime('2023-01-01').date()

        # Verify dataframes are displayed (mocked data)
        assert not at.session_state.financial_raw_data.empty
        assert at.dataframe[0].value.equals(mock_trends_raw_data['NKE'].head()) # Mock returns NKE data
        assert at.dataframe[1].value.equals(mock_financial_raw_data.head())


    def test_data_preprocessing_page_no_data_error():
        at = get_app_test_instance("2. Data Preprocessing & Feature Engineering")
        assert at.error[0].value == "Please fetch data in section '1. Data Acquisition' first."

    def test_data_preprocessing_page_process_data():
        at = get_app_test_instance("2. Data Preprocessing & Feature Engineering")
        # Manually set session state as if data was fetched
        at.session_state.trends_raw_data = mock_trends_raw_data
        at.session_state.financial_raw_data = mock_financial_raw_data
        at.run() # Rerun with session state updated

        at.number_input[0].set_value(24).run()
        at.button[0].click().run()

        assert at.session_state.ma_window == 24
        assert not at.session_state.processed_data == {}
        assert at.dataframe[0].value.equals(mock_processed_data['NKE'].head())
        assert at.dataframe[1].value.equals(mock_processed_data['NKE']['search_z'].describe().to_frame().T)


    def test_exploratory_visual_analysis_page_no_data_error():
        at = get_app_test_instance("3. Exploratory Visual Analysis")
        assert at.error[0].value == "Please process data in section '2. Data Preprocessing & Feature Engineering' first."

    def test_exploratory_visual_analysis_page_generate_plot():
        at = get_app_test_instance("3. Exploratory Visual Analysis")
        # Manually set session state as if data was processed
        at.session_state.processed_data = mock_processed_data
        at.run()

        at.button[0].click().run()
        assert at.session_state.plots_generated_exploratory is True
        assert at.image[0].caption == f'Google Trends Search Volume vs. Stock Performance for {at.session_state.selected_ticker}'

    def test_lead_lag_cross_correlation_page_no_data_error():
        at = get_app_test_instance("4. Lead-Lag Cross-Correlation Analysis")
        assert at.error[0].value == "Please process data in section '2. Data Preprocessing & Feature Engineering' first."

    def test_lead_lag_cross_correlation_page_generate_plot():
        at = get_app_test_instance("4. Lead-Lag Cross-Correlation Analysis")
        at.session_state.processed_data = mock_processed_data
        at.run()

        at.number_input[0].set_value(10).run()
        at.button[0].click().run()

        assert at.session_state.max_lag_corr == 10
        assert at.session_state.plots_generated_correlation is True
        assert at.image[0].caption == f'Lead-Lag Cross-Correlation for {at.session_state.selected_ticker}'

    def test_granger_causality_page_no_data_error():
        at = get_app_test_instance("5. Granger Causality Testing")
        assert at.error[0].value == "Please process data in section '2. Data Preprocessing & Feature Engineering' first."

    def test_granger_causality_page_perform_test():
        at = get_app_test_instance("5. Granger Causality Testing")
        at.session_state.processed_data = mock_processed_data
        at.run()

        # Mock sys.stdout for capturing print output
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            at.number_input[0].set_value(2).run()
            at.button[0].click().run()
            granger_output = mock_stdout.getvalue()

        assert at.session_state.granger_maxlag == 2
        # Verify that perform_granger_causality was called (it doesn't directly update a session state variable in the provided app code)
        # The app code assigns `buffer.getvalue()` to `st.session_state.granger_results_output`
        assert at.session_state.granger_results_output == granger_output # This might be empty if the mocked function doesn't write.
        assert "Granger Causality Test Results:" in at.markdown[at.markdown.index("### Granger Causality Test Results:")].value
        assert at.text[0].value == granger_output


    def test_prototype_signal_page_no_data_error():
        at = get_app_test_instance("6. Prototype Signal Construction & Performance Evaluation")
        assert at.error[0].value == "Please process data in section '2. Data Preprocessing & Feature Engineering' first."

    def test_prototype_signal_page_evaluate_performance():
        at = get_app_test_instance("6. Prototype Signal Construction & Performance Evaluation")
        at.session_state.processed_data = mock_processed_data
        at.run()

        at.button[0].click().run()

        assert not at.session_state.signal_performance_summary.empty
        assert at.session_state.plots_generated_signal is True
        assert at.image[0].caption == f'Cumulative Returns for {at.session_state.selected_ticker}'
        assert at.image[1].caption == f'Rolling IC for {at.session_state.selected_ticker}'
        assert at.image[2].caption == f'Scatter Plot for {at.session_state.selected_ticker}'
        assert at.dataframe[0].value.equals(mock_signal_performance_summary)


    def test_alternative_data_evaluation_framework_page():
        at = get_app_test_instance("7. Alternative Data Evaluation Framework")
        at.button[0].click().run()

        assert at.session_state.overall_alt_data_score is not None
        # Check that the scorecard DataFrame is displayed
        scorecard_df = pd.DataFrame.from_dict({k:v for k,v in mock_evaluation.items() if k != 'Dataset'}, orient='index', columns=['Score'])
        scorecard_df.index.name = 'Dimension'
        assert at.dataframe[0].value.equals(scorecard_df)
        assert at.markdown[at.markdown.index("**Overall Alt Data Score: 3.5 / 5.0**")].value == "**Overall Alt Data Score: 3.5 / 5.0**"
