import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from streamlit_lottie import st_lottie
import requests
from PIL import Image
import seaborn as sns
import io
import plotly.express as px
import plotly.figure_factory as ff

# ---------------------
# Functions
# ---------------------

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_data():
    data = pd.read_excel('data/data_Samsung_HW2.xlsx')
    return data

def outlier_detection(data_1d, quantile=0.975):
    y = data_1d.to_numpy().flatten()
    idx = np.isnan(y)
    y = y[~idx]
    unc_mean = np.mean(y)
    sigma = np.std(y, ddof=1)
    Z_crit = stats.norm.ppf(quantile)
    lower_bound = unc_mean - Z_crit * sigma
    upper_bound = unc_mean + Z_crit * sigma
    idx_outliers = (y < lower_bound) | (y > upper_bound)
    outlier_idx = np.where(idx_outliers)[0]
    return lower_bound, upper_bound, outlier_idx

# ---------------------
# Page Config
# ---------------------

st.set_page_config(
    page_title="📱 Samsung Price Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------
# Sidebar Navigation
# ---------------------

st.sidebar.title("📂 Navigation")
options = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dataset", "📈 Models", "🔍 Insights"])

# ---------------------
# Home Page
# ---------------------

if options == "🏠 Home":
    # Load Lottie Animation
    lottie_home = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
    st_lottie(lottie_home, key="home_anim", height=200)
    
    st.title("📱 Samsung Cell Phone Price Prediction")
    st.markdown("""
    Welcome to our **Samsung Cell Phone Price Prediction** app! 🎉

    This application showcases our data mining homework where we build and evaluate various regression models to predict the prices of Samsung cell phones based on multiple features.

    **Group Members:**
    - Luc Girel
    - Erwann Zarod-Wermeister
    - Marine Deflandre
    - Patrick Cheurfa

    Navigate through the sidebar to explore the dataset, models, and our insights.
    """)

    # Display a fun GIF
    st.markdown("![Fun GIF](https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif)")

# ---------------------
# Dataset Page
# ---------------------

elif options == "📊 Dataset":
    st.title("📊 Dataset Overview")

    data = load_data()

    st.subheader("🔍 First Five Rows of the Dataset")
    st.dataframe(data.head())

    st.subheader("📝 Dataset Information")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Identify and handle mixed-type columns
    mixed_type_cols = data.columns[data.apply(lambda col: col.map(type).nunique()) > 1]

    if len(mixed_type_cols) > 0:
        st.warning("⚠️ The following columns have mixed types and will be converted to numeric (non-convertible values will be set as NaN):")
        st.write(mixed_type_cols.tolist())
        
        # Convert mixed-type columns to numeric, forcing errors to NaN
        data[mixed_type_cols] = data[mixed_type_cols].apply(pd.to_numeric, errors='coerce')
        st.success("🔧 Mixed-type columns converted to numeric.")

    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    corr = numeric_data.corr()

    # Display the correlation heatmap
    st.subheader("🌐 Feature Correlations")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('🌐 Correlation Heatmap of Features')
    st.pyplot(fig)

    # Add a Lottie Animation
    lottie_dataset = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_touohxv0.json")
    st_lottie(lottie_dataset, key="dataset_anim", height=150)

# ---------------------
# Models Page
# ---------------------

elif options == "📈 Models":
    st.title("📈 Regression Models and Results")

    data = load_data()

    # Model 1: Basic Linear Regression
    st.markdown("### 🔍 Model 1: Basic Linear Regression")

    # Data Preparation
    data['Li-Ion'] = np.where(data['battery_type'] == 'Li-Ion', 1, 0)
    y = data['price(USD)']
    X = data[['Li-Ion', 'inches', 'battery', 'ram(GB)', 'weight(g)', 'storage(GB)', 'Resolution_product', 'days_from_release_date', 'up_to_8K']]
    X = sm.add_constant(X)

    # Fit Model 1
    with st.spinner('Training Model 1...'):
        model1 = sm.OLS(y, X).fit()
    st.success('Model 1 trained! 🎉')

    # Display OLS Summary
    st.subheader("📄 OLS Regression Results for Model 1")
    st.text(model1.summary())

    # Interpretation of Significant Parameters
    st.subheader("🔑 Interpretation of Significant Parameters")
    significant_vars = model1.pvalues[model1.pvalues < 0.05].index.tolist()
    if 'const' in significant_vars:
        significant_vars.remove('const')

    if significant_vars:
        for var in significant_vars:
            coef = model1.params[var]
            pval = model1.pvalues[var]
            st.markdown(f"- **{var}**: Coefficient = {coef:.4f} (p-value = {pval:.4f})")
    else:
        st.write("No significant variables at the 95% confidence level.")

    # Cross-Validation for Model 1
    st.subheader("🔄 Leave-One-Out Cross-Validation (LOO-CV) for Model 1")

    lm = LinearRegression()
    with st.spinner('Performing LOO-CV for Model 1...'):
        scores_loo_SE_M1 = cross_val_score(lm, X, y, scoring='neg_mean_squared_error', cv=LeaveOneOut())
        scores_loo_AE_M1 = np.sqrt(-scores_loo_SE_M1)
        RMSE = np.sqrt(np.mean(-scores_loo_SE_M1))
        MAE = np.mean(scores_loo_AE_M1)
    st.success('LOO-CV for Model 1 completed! 🎉')

    # Display Metrics
    st.markdown("### 📊 Model 1 Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔺 RMSE", f"{RMSE:.2f}")
    with col2:
        st.metric("🔻 MAE", f"{MAE:.2f}")

    # Interactive Plotly Scatter Plot for Model 1
    st.markdown("### 🔍 Actual vs Predicted Prices for Model 1")
    lm.fit(X, y)
    predictions = lm.predict(X)

    plot_df = pd.DataFrame({
        'Actual Price (USD)': y,
        'Predicted Price (USD)': predictions
    })

    fig = px.scatter(plot_df, 
                     x='Actual Price (USD)', 
                     y='Predicted Price (USD)', 
                     trendline='ols',
                     title="📈 Actual vs Predicted Prices",
                     labels={'Actual Price (USD)': 'Actual Price', 'Predicted Price (USD)': 'Predicted Price'},
                     template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Example for Model 2: Including Interaction and Quadratic Terms
    st.markdown("### 🔄 Model 2: Including Interaction and Quadratic Terms")

    # Generate Polynomial Features for Model 2
    x2 = data[['inches', 'battery', 'ram(GB)', 'weight(g)', 'storage(GB)', 'Resolution_product', 'days_from_release_date', 'up_to_8K']]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(x2)
    X_poly = sm.add_constant(X_poly)

    # Fit Model 2
    with st.spinner('Training Model 2...'):
        model2 = sm.OLS(y, X_poly).fit()
    st.success('Model 2 trained! 🎉')

    # Display OLS Summary for Model 2
    st.subheader("📄 OLS Regression Results for Model 2")
    st.text(model2.summary())

    # Cross-Validation for Model 2
    st.subheader("🔄 Leave-One-Out Cross-Validation (LOO-CV) for Model 2")

    with st.spinner('Performing LOO-CV for Model 2...'):
        scores_loo_SE_M2 = cross_val_score(lm, X_poly, y, scoring='neg_mean_squared_error', cv=LeaveOneOut())
        scores_loo_AE_M2 = np.sqrt(-scores_loo_SE_M2)
        RMSE2 = np.sqrt(np.mean(-scores_loo_SE_M2))
        MAE2 = np.mean(scores_loo_AE_M2)
    st.success('LOO-CV for Model 2 completed! 🎉')

    # Display Metrics for Models 1 and 2
    st.markdown("### 📊 Comparison of Models")
    comparison_df = pd.DataFrame({
        'Model': ['Model 1', 'Model 2'],
        'RMSE': [RMSE, RMSE2],
        'MAE': [MAE, MAE2]
    })
    st.table(comparison_df)

    # Repeat similar steps for Models 3, 4, and 5 with enhancements...

# ---------------------
# Insights Page
# ---------------------

elif options == "🔍 Insights":
    st.title("🔍 Insights and Discussion")

    st.markdown("""
    ### 📈 Comparison of Models

    | Model   | RMSE        | MAE         |
    |---------|-------------|-------------|
    | Model 1 | 239.87      | 158.04      |
    | Model 2 | 371.96      | 217.03      |
    | Model 3 | 0.56        | 0.43        |
    | Model 4 | 188.32      | 138.49      |
    | Model 5 | 0.53        | 0.42        |

    **Insights:**

    - **Model 1** serves as the baseline with reasonable performance.
    - **Model 2** shows worse performance, likely due to overfitting from adding too many interaction and quadratic terms.
    - **Model 3** and **Model 5** utilize log transformation, which significantly reduces RMSE and MAE, indicating better predictive performance.
    - **Model 4** shows improvement by removing outliers and selecting key features, achieving lower RMSE and MAE compared to Model 1.

    ### 📊 Visual Interpretations

    - **Actual vs Predicted Prices:** Demonstrates how well the model predicts the actual prices.
    - **Residuals Plot:** Helps in diagnosing the fit of the model.
    - **Correlation Heatmap:** Shows the relationships between different features.

    ### 🧩 Conclusion

    The refined models, especially after removing outliers and selecting significant features, offer improved predictive performance. Log transformations further enhance the model by stabilizing variance and normalizing the distribution of the dependent variable.
    """)

    # Add a fun conclusion GIF
    st.markdown("![Conclusion GIF](https://media.giphy.com/media/l0HlSNOxJB956qwfK/giphy.gif)")

    # Add a Lottie animation for conclusion
    lottie_conclusion = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
    st_lottie(lottie_conclusion, key="conclusion_anim", height=150)