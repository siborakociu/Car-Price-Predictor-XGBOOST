import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 3rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .price-text {
        font-size: 5.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin: 0;
        padding: 0;
    }
    .price-label {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved files
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('xgb_car_price_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        categorical_values = joblib.load('categorical_values.pkl')
        df_sample = joblib.load('df_sample.pkl')
        return model, encoders, categorical_values, df_sample
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.error("Please ensure all .pkl files are in the same directory as this script.")
        return None, None, None, None

model, encoders, categorical_values, df_sample = load_model_and_data()

if model is None:
    st.stop()

# Header
st.markdown('<p class="main-header">🚗 Car Price Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Price Predictor", "Data Exploration", "Model Performance"])

# PAGE 1: PRICE PREDICTOR
if page == "Price Predictor":
    st.markdown('<p class="sub-header">🎯 Predict Your Car\'s Market Price</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vehicle Details")
        
        year = st.slider("Year", 
                        min_value=1990, 
                        max_value=2026, 
                        value=2020,
                        help="Manufacturing year of the vehicle")
        
        make = st.selectbox("Make", 
                           options=categorical_values['makes'],
                           help="Vehicle manufacturer")
        
        # Filter models based on selected make
        available_models = df_sample[df_sample['make'] == make]['model'].unique().tolist()
        if len(available_models) == 0:
            available_models = categorical_values['models']
        
        model_select = st.selectbox("Model", 
                                   options=sorted(available_models),
                                   help="Vehicle model")
        
        body = st.selectbox("Body Type", 
                           options=categorical_values['body_types'],
                           help="Type of vehicle body")
    
    with col2:
        st.subheader("Condition & Mileage")
        
        condition = st.slider("Condition", 
                             min_value=1.0, 
                             max_value=49.0, 
                             value=25.0,
                             step=1.0,
                             help="Vehicle condition score (1-49, higher is better)")
        
        odometer = st.number_input("Odometer (miles)", 
                                   min_value=1, 
                                   max_value=500000, 
                                   value=50000,
                                   step=1000,
                                   help="Total miles driven")
        
        transmission = st.selectbox("Transmission", 
                                   options=categorical_values['transmissions'],
                                   help="Type of transmission")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Price", use_container_width=True):
        try:
            # Encode categorical variables
            make_encoded = encoders['make'].transform([make])[0]
            model_encoded = encoders['model'].transform([model_select])[0]
            body_encoded = encoders['body'].transform([body])[0]
            transmission_encoded = encoders['transmission'].transform([transmission])[0]
            
            # Create feature array
            features = np.array([[
                year,
                condition,
                odometer,
                make_encoded,
                model_encoded,
                body_encoded,
                transmission_encoded
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<p class="price-label">Predicted Market Price</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="price-text">${prediction:,.2f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please ensure all selections are valid.")

# PAGE 2: DATA EXPLORATION
elif page == "Data Exploration":
    st.markdown('<p class="sub-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df_sample):,}")
    with col2:
        st.metric("Unique Makes", len(df_sample['make'].unique()))
    with col3:
        st.metric("Unique Models", len(df_sample['model'].unique()))
    with col4:
        st.metric("Avg Price", f"${df_sample['sellingprice'].mean():,.0f}")
    
    st.markdown("---")
    
    # Categorical features visualization
    st.subheader("Categorical Feature Distributions")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Body Type Distribution
    body_counts = df_sample['body'].value_counts()
    axes[0,0].bar(body_counts.index, body_counts.values, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Body Type Distribution', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Body Type')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Transmission Distribution
    trans_counts = df_sample['transmission'].value_counts()
    axes[0,1].bar(trans_counts.index, trans_counts.values, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('Transmission Distribution', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Transmission')
    axes[0,1].set_ylabel('Count')
    
    # Top 10 Makes
    top_makes = df_sample['make'].value_counts().head(10)
    axes[1,0].barh(top_makes.index, top_makes.values, color='lightgreen', edgecolor='black')
    axes[1,0].set_title('Top 10 Makes', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Count')
    axes[1,0].invert_yaxis()
    
    # Top 10 Models
    top_models = df_sample['model'].value_counts().head(10)
    axes[1,1].barh(top_models.index, top_models.values, color='plum', edgecolor='black')
    axes[1,1].set_title('Top 10 Models', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Count')
    axes[1,1].invert_yaxis()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Numerical features
    st.subheader("Numerical Feature Distributions")
    
    num_cols = ['year', 'condition', 'odometer', 'sellingprice']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(num_cols):
        axes[idx].hist(df_sample[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        axes[idx].set_title(f'{col.title()} Distribution', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel(col.title())
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df_sample[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    st.pyplot(fig)
    
    # Sample data preview
    st.markdown("---")
    st.subheader("Sample Data Preview")
    st.dataframe(df_sample.head(100), use_container_width=True)

# PAGE 3: MODEL PERFORMANCE
elif page == "Model Performance":
    st.markdown('<p class="sub-header">🎯 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    # Calculate predictions on sample data for visualization
    feature_cols = ['year', 'condition', 'odometer', 'make_encoded', 
                   'model_encoded', 'body_encoded', 'transmission_encoded']
    
    X = df_sample[feature_cols]
    y = df_sample['sellingprice']
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}", 
                 help="Proportion of variance explained by the model (closer to 1 is better)")
    with col2:
        st.metric("MAE", f"${mae:,.2f}", 
                 help="Mean Absolute Error - average prediction error")
    with col3:
        st.metric("RMSE", f"${rmse:,.2f}", 
                 help="Root Mean Squared Error - penalizes large errors more")
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("Feature Importance Analysis")
    
    importance_scores = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': importance_scores.keys(),
        'Importance': importance_scores.values()
    })
    importance_df['Importance (%)'] = (importance_df['Importance'] / 
                                       importance_df['Importance'].sum()) * 100
    importance_df = importance_df.sort_values(by='Importance (%)', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance (%)'], 
                   color='steelblue', edgecolor='black')
    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_title("XGBoost Feature Importance", fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'{width:.1f}%', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Car Price Prediction System | Powered by XGBoost</p>
        <p>Model trained on 99,970 vehicle sales records</p>
    </div>
""", unsafe_allow_html=True)
