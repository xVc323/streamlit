# app.py

import streamlit as st
from google.cloud import bigquery
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency, ttest_ind
import json
import os
from dotenv import load_dotenv

# =========================
# Configuration and Setup
# =========================

# Set Streamlit page configuration
st.set_page_config(page_title="January 2021 Web Analytics Dashboard", layout="wide")

# Title of the dashboard
st.title("📊 January 2021 Web Analytics Dashboard")

# =========================
# Load Environment Variables
# =========================

# Load environment variables from .env file
load_dotenv()

# Get the path to the credentials from environment variables
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not credentials_path:
    st.error("GOOGLE_APPLICATION_CREDENTIALS not found in environment variables.")
    st.stop()

# Set the environment variable for Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Initialize BigQuery client
client = bigquery.Client()

# Function to run SQL queries and return a DataFrame
@st.cache_data
def run_query(query):
    try:
        query_job = client.query(query)
        return query_job.to_dataframe()
    except Exception as e:
        st.error(f"An error occurred while running the query: {e}")
        return pd.DataFrame()

# =========================
# Business Problem Functions
# =========================

def problem1_conversion_by_time():
    query = """
    -- Problem 1: Impact of Time of Day on Conversion Rate
    WITH hourly_data AS (
        SELECT
            EXTRACT(HOUR FROM TIMESTAMP_MICROS(event_timestamp)) AS hour,
            COUNT(DISTINCT user_pseudo_id) AS total_users,
            COUNT(DISTINCT CASE WHEN event_name = 'purchase' THEN user_pseudo_id END) AS converted_users
        FROM
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_202101*`
        WHERE
            event_name IN ('session_start', 'purchase')
        GROUP BY
            hour
    )
    
    SELECT
        hour,
        total_users,
        converted_users,
        SAFE_DIVIDE(converted_users, total_users) AS conversion_rate
    FROM
        hourly_data
    ORDER BY
        hour
    """
    df = run_query(query)
    return df

def problem2_category_conversion():
    query = """
    -- Problem 2: Product Categories with Highest Add-to-Cart to Order Rate
    WITH category_actions AS (
        SELECT
            (SELECT value.string_value FROM UNNEST(event_params)
                WHERE key = 'item_category') AS category,
            COUNTIF(event_name = 'add_to_cart') AS add_to_cart_count,
            COUNTIF(event_name = 'purchase') AS purchase_count
        FROM
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_202101*`
        GROUP BY
            category
    )
    
    SELECT
        category,
        add_to_cart_count,
        purchase_count,
        SAFE_DIVIDE(purchase_count, add_to_cart_count) AS conversion_rate
    FROM
        category_actions
    WHERE
        category IS NOT NULL
    ORDER BY
        conversion_rate DESC
    """
    df = run_query(query)
    # Handle missing categories
    df['category'] = df['category'].fillna('Unknown')
    return df

def problem3_user_type_conversion():
    query = """
    -- Problem 3: Conversion Rates Between New and Returning Users
    WITH user_segments AS (
        SELECT
            user_pseudo_id,
            MIN(event_timestamp) AS first_event
        FROM
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_202101*`
        GROUP BY
            user_pseudo_id
    ),
    conversion_data AS (
        SELECT
            CASE 
                WHEN e.event_timestamp = us.first_event THEN 'New'
                ELSE 'Returning'
            END AS user_type,
            COUNT(DISTINCT e.user_pseudo_id) AS total_users,
            COUNT(DISTINCT CASE WHEN e.event_name = 'purchase' THEN e.user_pseudo_id END) AS converted_users
        FROM
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_202101*` e
        JOIN
            user_segments us
        ON
            e.user_pseudo_id = us.user_pseudo_id
        WHERE
            e.event_name IN ('session_start', 'purchase')
        GROUP BY
            user_type
    )
    
    SELECT
        user_type,
        total_users,
        converted_users,
        SAFE_DIVIDE(converted_users, total_users) AS conversion_rate
    FROM
        conversion_data
    """
    df = run_query(query)
    return df

def problem4_seasonality_sales():
    query = """
    -- Problem 4: Seasonality and Its Effect on Sales
    WITH daily_sales AS (
        SELECT
            DATE(TIMESTAMP_MICROS(event_timestamp)) AS date,
            COUNTIF(event_name = 'purchase') AS total_purchases,
            SUM(CAST((SELECT value.int_value FROM UNNEST(event_params)
                WHERE key = 'value') AS FLOAT64)) AS total_revenue
        FROM
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_202101*`
        WHERE
            event_name = 'purchase'
        GROUP BY
            date
    )
    
    SELECT
        date,
        total_purchases,
        total_revenue,
        AVG(total_purchases) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS moving_avg_purchases
    FROM
        daily_sales
    ORDER BY
        date
    """
    df = run_query(query)
    return df

# =========================
# Statistical Test Functions
# =========================

def chi_square_test(df):
    if df.empty:
        st.error("DataFrame is empty. Cannot perform Chi-Square Test.")
        return None, None
    # Create a contingency table
    contingency_table = pd.crosstab(df['hour'], [df['converted_users'], df['total_users'] - df['converted_users']])
    # Perform Chi-Square Test
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    return chi2, p

def t_test_new_returning():
    # User-level conversion rates for T-Test
    query = """
    -- Adjusted Problem 3: User-Level Conversion Rates for T-Test
    WITH user_segments AS (
        SELECT
            user_pseudo_id,
            MIN(event_timestamp) AS first_event
        FROM
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_202101*`
        GROUP BY
            user_pseudo_id
    ),
    user_conversions AS (
        SELECT
            e.user_pseudo_id,
            CASE 
                WHEN e.event_timestamp = us.first_event THEN 'New'
                ELSE 'Returning'
            END AS user_type,
            COUNTIF(e.event_name = 'purchase') AS purchases
        FROM
            `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_202101*` e
        JOIN
            user_segments us
        ON
            e.user_pseudo_id = us.user_pseudo_id
        GROUP BY
            e.user_pseudo_id, user_type
    )
    
    SELECT
        user_type,
        purchases
    FROM
        user_conversions
    """
    user_df = run_query(query)
    
    if user_df.empty:
        st.error("User conversion data is empty. Cannot perform T-Test.")
        return None, None
    
    # Calculate conversion rate per user (binary: converted or not)
    user_df['converted'] = user_df['purchases'].apply(lambda x: 1 if x > 0 else 0)
    
    # Split into groups
    new_users = user_df[user_df['user_type'] == 'New']['converted']
    returning_users = user_df[user_df['user_type'] == 'Returning']['converted']
    
    # Check if both groups have data
    if new_users.empty or returning_users.empty:
        st.error("One of the user groups is empty. Cannot perform T-Test.")
        return None, None
    
    # Perform T-Test
    t_stat, p_val = ttest_ind(new_users, returning_users, equal_var=False)
    return t_stat, p_val

# =========================
# Streamlit Sidebar Navigation
# =========================

st.sidebar.title("🔍 Navigation")
options = st.sidebar.selectbox("Select a Business Problem", 
                               ["Conversion Rate by Time of Day",
                                "Category Conversion Rates",
                                "User Type Conversion Rates",
                                "Seasonality in Sales"])

# =========================
# Problem 1: Conversion Rate by Time of Day
# =========================

if options == "Conversion Rate by Time of Day":
    st.header("📈 Impact of Time of Day on Conversion Rate")
    st.markdown("### Objective:")
    st.write("Determine if the time of day influences the conversion rate.")
    
    # Fetch data
    with st.spinner("Fetching data..."):
        df1 = problem1_conversion_by_time()
    
    if not df1.empty:
        # Display DataFrame
        st.subheader("Conversion Rates by Hour")
        st.dataframe(df1)
        
        # Visualization
        fig1 = px.line(df1, x='hour', y='conversion_rate',
                      title='Conversion Rate by Hour of Day',
                      labels={'hour': 'Hour of Day', 'conversion_rate': 'Conversion Rate'})
        fig1.update_traces(mode='markers+lines')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Statistical Test
        st.subheader("Chi-Square Test for Independence")
        chi2, p = chi_square_test(df1)
        if chi2 is not None and p is not None:
            st.write(f"**Chi2 Statistic:** {chi2:.4f}")
            st.write(f"**P-Value:** {p:.4f}")
            if p < 0.05:
                st.success("Result: Reject the null hypothesis. Time of day **influences** conversion rate.")
            else:
                st.warning("Result: Fail to reject the null hypothesis. No significant influence of time of day on conversion rate.")
    else:
        st.error("No data available for this analysis.")

# =========================
# Problem 2: Category Conversion Rates
# =========================

elif options == "Category Conversion Rates":
    st.header("🛒 Product Categories with Highest Add-to-Cart to Order Rate")
    st.markdown("### Objective:")
    st.write("Identify which product categories have the highest rate of converting add-to-cart actions into orders.")
    
    # Fetch data
    with st.spinner("Fetching data..."):
        df2 = problem2_category_conversion()
    
    if not df2.empty:
        # Display DataFrame
        st.subheader("Conversion Rates by Product Category")
        st.dataframe(df2)
        
        # Visualization
        fig2 = px.bar(df2, x='category', y='conversion_rate',
                     title='Add-to-Cart to Order Conversion Rate by Category',
                     labels={'category': 'Product Category', 'conversion_rate': 'Conversion Rate'},
                     hover_data=['add_to_cart_count', 'purchase_count'])
        fig2.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("No data available for this analysis.")

# =========================
# Problem 3: User Type Conversion Rates
# =========================

elif options == "User Type Conversion Rates":
    st.header("👥 Conversion Rates: New vs. Returning Users")
    st.markdown("### Objective:")
    st.write("Analyze whether returning users have higher conversion rates compared to new users.")
    
    # Fetch data
    with st.spinner("Fetching data..."):
        df3 = problem3_user_type_conversion()
    
    if not df3.empty:
        # Display DataFrame
        st.subheader("Conversion Rates by User Type")
        st.dataframe(df3)
        
        # Visualization
        fig3 = px.box(df3, x='user_type', y='conversion_rate',
                     title='Conversion Rates: New vs. Returning Users',
                     labels={'user_type': 'User Type', 'conversion_rate': 'Conversion Rate'})
        st.plotly_chart(fig3, use_container_width=True)
        
        # Statistical Test
        st.subheader("T-Test for Difference in Conversion Rates")
        t_stat, p_val = t_test_new_returning()
        if t_stat is not None and p_val is not None:
            st.write(f"**T-Statistic:** {t_stat:.4f}")
            st.write(f"**P-Value:** {p_val:.4f}")
            if p_val < 0.05:
                st.success("Result: Reject the null hypothesis. Significant difference in conversion rates between new and returning users.")
            else:
                st.warning("Result: Fail to reject the null hypothesis. No significant difference in conversion rates between new and returning users.")
    else:
        st.error("No data available for this analysis.")

# =========================
# Problem 4: Seasonality in Sales
# =========================

elif options == "Seasonality in Sales":
    st.header("📅 Seasonality and Its Effect on Sales")
    st.markdown("### Objective:")
    st.write("Examine if there are any weekly patterns or trends in sales throughout January 2021.")
    
    # Fetch data
    with st.spinner("Fetching data..."):
        df4 = problem4_seasonality_sales()
    
    if not df4.empty:
        # Display DataFrame
        st.subheader("Daily Sales and 7-Day Moving Average")
        st.dataframe(df4)
        
        # Visualization
        fig4 = px.line(df4, x='date', y=['total_purchases', 'moving_avg_purchases'],
                      title='Daily Purchases and 7-Day Moving Average',
                      labels={'date': 'Date', 'value': 'Number of Purchases'},
                      hover_data={'date': '|%B %d, %Y'})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.error("No data available for this analysis.")

# =========================
# Footer
# =========================

st.markdown("---")
st.markdown("© 2024 | Web Analytics Dashboard 🎉")
