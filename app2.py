import streamlit as st
from sklearn import datasets
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Interactive Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("📊 Interactive Data Explorer")
st.markdown(
    """
    Explore different datasets using interactive visualizations.
    - **Python libraries:** Streamlit, Pandas, Scikit-learn, Plotly
    - **Features:** Interactive widgets, data caching, dynamic plots
    """
)

# Sidebar for user input
st.sidebar.header("User Input Features")

# Function to load datasets
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

# Select dataset
dataset_name = st.sidebar.selectbox(
    "Select Dataset", ("Iris", "Wine", "Breast Cancer")
)
data = load_data(dataset_name)

# Show dataset
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data.head())

# Select features for x and y axes
st.sidebar.subheader("Plot Settings")
x_axis = st.sidebar.selectbox("X Axis", options=data.columns)
y_axis = st.sidebar.selectbox("Y Axis", options=data.columns)

# Color option
color_option = st.sidebar.selectbox("Color By", options=data.columns)

# Plotting
fig = px.scatter(
    data_frame=data,
    x=x_axis,
    y=y_axis,
    color=color_option,
    title=f"{y_axis} vs {x_axis} ({dataset_name} Dataset)",
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)