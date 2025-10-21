import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('diamond_price_pipeline.pkl')
encoder = joblib.load('diamond_categorical_encoder.pkl')
categorical_choices = dict(zip(encoder.feature_names_in_, encoder.categories_))


# Streamlit app
st.title('💎 Predicting the Price of Diamond')
st.write('Predict the price of diamond based on the given dataset. Link: [Predicting the Price of Diamond](https://www.kaggle.com/competitions/predicting-the-price-of-diamond)')

# Input features
col1, col2 = st.columns(2)

with col1:
    st.subheader("💎 Diamond Characteristics")
    
    # Carat (weight)
    carat = st.slider(
        "Carat (Weight)", 
        min_value=0.2, 
        max_value=5.0, 
        value=1.0, 
        step=0.1,
        help="Diamond weight in carats"
    )
    
    # Cut quality
    cut = st.selectbox(
        "Cut Quality",
        options=categorical_choices['cut'],
        help="Quality of the cut (affects brilliance)"
    )
    
    # Color grade
    color = st.selectbox(
        "Color Grade",
        options=categorical_choices['color'],
        help="Color grade from D (best) to J (worst)"
    )
    
    # Clarity grade
    clarity = st.selectbox(
        "Clarity Grade",
        options=categorical_choices['clarity'],
        help="Clarity grade from IF (best) to I1 (worst)"
    )

with col2:
    st.subheader("📏 Physical Dimensions")
    
    # Table percentage
    table = st.slider(
        "Table %", 
        min_value=43.0, 
        max_value=95.0, 
        value=57.0, 
        step=0.1,
        help="Table width relative to average diameter"
    )
    
    # X dimension (length)
    x = st.slider(
        "Length (X)", 
        min_value=3.0, 
        max_value=10.0, 
        value=5.0, 
        step=0.01,
        help="Length in mm"
    )
    
    # Y dimension (width)
    y = st.slider(
        "Width (Y)", 
        min_value=3.0, 
        max_value=10.0, 
        value=5.0, 
        step=0.01,
        help="Width in mm"
    )
    
    # Z dimension (height)
    z = st.slider(
        "Height (Z)", 
        min_value=1.0, 
        max_value=7.0, 
        value=3.0, 
        step=0.01,
        help="Height in mm"
    )


# Make prediction
if st.button("🔮 Predict Diamond Price", type="primary", use_container_width=True):
    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([{
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }])

    # Encode categorical variables in the correct order (cut, clarity, color)
    input_df[['cut', 'clarity', 'color']] = encoder.transform(input_df[['cut', 'clarity', 'color']])
    prediction = pipeline.predict(input_df)[0]
    
    st.success(f"**Predicted Price: ${prediction:,.2f}**")
    
    # Display input summary
    st.subheader("📋 Input Summary")
    summary_data = {
        'Carat': f"{carat} ct",
        'Cut': cut,
        'Color': color,
        'Clarity': clarity,
        'Table %': f"{table}%",
        'Dimensions': f"{x} × {y} × {z} mm"
    }
    
    for key, value in summary_data.items():
        st.write(f"**{key}:** {value}")