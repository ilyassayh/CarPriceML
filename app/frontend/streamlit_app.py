"""
Streamlit Frontend for CarPriceML

This is the user interface for the CarPriceML application.
It provides a web-based form for users to input car details and get price predictions.

The frontend communicates with the FastAPI backend (app/api/main.py) via HTTP requests.
The backend API must be running (typically on http://localhost:8000) for this app to work.

Communication Flow:
    1. User fills out the form with car features
    2. Frontend sends POST request to {API_URL}/predict endpoint
    3. Backend processes the request and returns predicted price
    4. Frontend displays the result to the user
"""

# Standard library imports
import json
import os
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import pandas as pd
import requests
import streamlit as st

# Define paths relative to the project root
# APP_ROOT is two levels up from this file (app/frontend/ -> project root)
APP_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = APP_ROOT / 'models'
META_PATH = MODELS_DIR / 'metadata.json'

# Get API URL from environment variable, default to localhost
# This points to the FastAPI backend (app/api/main.py)
# The backend must be running for this frontend to function
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Configure Streamlit page settings
st.set_page_config(page_title='CarPriceML', page_icon='🚗', layout='centered')
st.title('🚗 CarPriceML — Estimation du prix')
st.markdown('*Estimez le prix de votre véhicule en quelques clics*')

# Load model metadata to understand required features
# Metadata contains information about numeric and categorical features expected by the model
metadata: Dict[str, Any] = {}
if META_PATH.exists():
    metadata = json.loads(META_PATH.read_text(encoding='utf-8'))

# Extract feature lists from metadata
numeric_cols = metadata.get('numeric_features', [])
categorical_cols = metadata.get('categorical_features', [])
expected_cols = numeric_cols + categorical_cols

# Define predefined options for all fields based on common car data
CATEGORICAL_OPTIONS = {
    'company': ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Skoda', 'Renault', 'Mahindra', 
                'Tata', 'Volkswagen', 'BMW', 'Mercedes-Benz', 'Audi', 'Jeep', 'Nissan', 
                'Datsun', 'Chevrolet', 'Fiat', 'MG', 'Volvo', 'Lexus', 'Jaguar', 'Land Rover', 'Daewoo'],
    'fuel': ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'],
    'transmission': ['Manual', 'Automatic'],
    'owner': ['First', 'Second', 'Third', 'Fourth & Above'],
    'seller_type': ['Individual', 'Dealer', 'Trustmark Dealer']
}

# Common model names (can be filtered by company if needed)
COMMON_MODELS = ['Swift', 'City', 'i20', 'Alto', 'Wagon', 'Baleno', 'Creta', 'Verna', 
                 'Fortuner', 'Innova', 'Corolla', 'Etios', 'Rapid', 'Duster', 'Figo', 
                 'EcoSport', 'X1', 'A6', 'Q5', 'C-Class', 'E-Class', 'S-Class']

# Common editions/trims
COMMON_EDITIONS = ['VDI', 'VXI', 'ZXI', 'LXI', 'Asta', 'Sportz', 'Magna', 'SX', 
                   'VX', 'ZX', 'Delta', 'Zeta', 'Alpha', 'Highline', 'Comfortline', 
                   'Trendline', 'Elegance', 'Ambition', 'Plus', 'Option']

# Numeric field options (ranges based on common car data)
YEAR_OPTIONS = list(range(2000, 2026))  # 2000 to 2025
KM_DRIVEN_OPTIONS = [0, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 
                     60000, 70000, 80000, 90000, 100000, 120000, 140000, 150000, 
                     170000, 200000]
ENGINE_CC_OPTIONS = [796, 998, 1061, 1086, 1197, 1198, 1199, 1248, 1298, 1364, 
                     1396, 1399, 1461, 1462, 1497, 1498, 1591, 1598, 1797, 1798, 
                     1968, 1995, 1998, 1999, 2143, 2179, 2362, 2477, 2494, 2755, 
                     2982, 2987, 3604]
POWER_BHP_OPTIONS = [34, 37, 40, 46, 47, 53, 57, 60, 64, 67, 68, 73, 74, 75, 78, 
                     81, 82, 83, 86, 88, 90, 93, 98, 100, 103, 108, 116, 117, 120, 
                     121, 126, 138, 141, 147, 155, 160, 168, 170, 171, 174, 177, 
                     183, 187, 188, 189, 190, 214, 241, 254, 280, 400]
TORQUE_NM_OPTIONS = [59, 60, 62, 69, 71, 76, 78, 84, 90, 91, 95, 96, 99, 104, 110, 
                     112, 113, 114, 115, 124, 130, 138, 145, 146, 151, 153, 160, 170, 
                     173, 177, 190, 200, 205, 213, 219, 224, 248, 250, 259, 260, 280, 
                     290, 300, 320, 343, 347, 360, 380, 382, 400, 430, 490, 500, 550, 
                     620, 640]
MILEAGE_MPG_OPTIONS = [22, 26, 28, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
                       43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
                       58, 59, 61, 62, 64, 66, 98]
SEATS_OPTIONS = [4, 5, 6, 7, 8]

# Create a simple form for user input with all selectboxes
with st.form('car_form'):
    st.subheader('📋 Informations du véhicule')
    inputs: Dict[str, Any] = {}
    
    # Organize fields in columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic car information with selectboxes
        if 'company' in categorical_cols:
            company_options = CATEGORICAL_OPTIONS.get('company', [])
            inputs['company'] = st.selectbox('Marque', options=['Sélectionner...'] + company_options, index=0)
            if inputs['company'] == 'Sélectionner...':
                inputs['company'] = None
        
        if 'model' in categorical_cols:
            inputs['model'] = st.selectbox('Modèle', options=['Sélectionner...'] + COMMON_MODELS, index=0)
            if inputs['model'] == 'Sélectionner...':
                inputs['model'] = None
        
        if 'edition' in categorical_cols:
            inputs['edition'] = st.selectbox('Édition', options=['Sélectionner...'] + COMMON_EDITIONS, index=0)
            if inputs['edition'] == 'Sélectionner...':
                inputs['edition'] = None
        
        if 'name' in categorical_cols:
            # Name can be a combination, so we'll allow text input but with a placeholder
            inputs['name'] = st.text_input('Nom complet du véhicule', value='', 
                                          help='Ex: Maruti Swift Dzire VDI', 
                                          placeholder='Laissez vide ou entrez le nom complet')
        
        # Year and mileage with selectboxes
        if 'year' in numeric_cols:
            inputs['year'] = st.selectbox('Année', options=['Sélectionner...'] + YEAR_OPTIONS, index=0)
            if inputs['year'] == 'Sélectionner...':
                inputs['year'] = None
        
        if 'km_driven' in numeric_cols:
            km_options_str = ['Sélectionner...'] + [f"{x:,} km" for x in KM_DRIVEN_OPTIONS]
            selected_km = st.selectbox('Kilométrage', options=km_options_str, index=0)
            if selected_km != 'Sélectionner...':
                inputs['km_driven'] = int(selected_km.replace(' km', '').replace(',', ''))
            else:
                inputs['km_driven'] = None
    
    with col2:
        # Engine specifications with selectboxes
        if 'engine_cc' in numeric_cols:
            engine_options_str = ['Sélectionner...'] + [f"{x} cc" for x in ENGINE_CC_OPTIONS]
            selected_engine = st.selectbox('Cylindrée', options=engine_options_str, index=0)
            if selected_engine != 'Sélectionner...':
                inputs['engine_cc'] = float(selected_engine.replace(' cc', ''))
            else:
                inputs['engine_cc'] = None
        
        if 'max_power_bhp' in numeric_cols:
            power_options_str = ['Sélectionner...'] + [f"{x} bhp" for x in POWER_BHP_OPTIONS]
            selected_power = st.selectbox('Puissance max', options=power_options_str, index=0)
            if selected_power != 'Sélectionner...':
                inputs['max_power_bhp'] = float(selected_power.replace(' bhp', ''))
            else:
                inputs['max_power_bhp'] = None
        
        if 'torque_nm' in numeric_cols:
            torque_options_str = ['Sélectionner...'] + [f"{x} Nm" for x in TORQUE_NM_OPTIONS]
            selected_torque = st.selectbox('Couple', options=torque_options_str, index=0)
            if selected_torque != 'Sélectionner...':
                inputs['torque_nm'] = float(selected_torque.replace(' Nm', ''))
            else:
                inputs['torque_nm'] = None
        
        if 'mileage_mpg' in numeric_cols:
            mileage_options_str = ['Sélectionner...'] + [f"{x} mpg" for x in MILEAGE_MPG_OPTIONS]
            selected_mileage = st.selectbox('Consommation', options=mileage_options_str, index=0)
            if selected_mileage != 'Sélectionner...':
                inputs['mileage_mpg'] = float(selected_mileage.replace(' mpg', ''))
            else:
                inputs['mileage_mpg'] = None
        
        if 'seats' in numeric_cols:
            seats_options_str = ['Sélectionner...'] + [f"{x} places" for x in SEATS_OPTIONS]
            selected_seats = st.selectbox('Nombre de places', options=seats_options_str, index=0)
            if selected_seats != 'Sélectionner...':
                inputs['seats'] = float(selected_seats.replace(' places', ''))
            else:
                inputs['seats'] = None
    
    # Categorical fields with selectboxes
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        if 'fuel' in categorical_cols:
            fuel_options = CATEGORICAL_OPTIONS.get('fuel', [])
            inputs['fuel'] = st.selectbox('Carburant', options=['Sélectionner...'] + fuel_options, index=0)
            if inputs['fuel'] == 'Sélectionner...':
                inputs['fuel'] = None
        
        if 'transmission' in categorical_cols:
            trans_options = CATEGORICAL_OPTIONS.get('transmission', [])
            inputs['transmission'] = st.selectbox('Transmission', options=['Sélectionner...'] + trans_options, index=0)
            if inputs['transmission'] == 'Sélectionner...':
                inputs['transmission'] = None
    
    with col4:
        if 'owner' in categorical_cols:
            owner_options = CATEGORICAL_OPTIONS.get('owner', [])
            inputs['owner'] = st.selectbox('Propriétaire', options=['Sélectionner...'] + owner_options, index=0)
            if inputs['owner'] == 'Sélectionner...':
                inputs['owner'] = None
        
        if 'seller_type' in categorical_cols:
            seller_options = CATEGORICAL_OPTIONS.get('seller_type', [])
            inputs['seller_type'] = st.selectbox('Type de vendeur', options=['Sélectionner...'] + seller_options, index=0)
            if inputs['seller_type'] == 'Sélectionner...':
                inputs['seller_type'] = None
    
    # Handle any remaining categorical fields that weren't in the options
    for col in categorical_cols:
        if col not in inputs:
            inputs[col] = st.selectbox(col.replace('_', ' ').title(), 
                                      options=['Sélectionner...', 'Autre'], index=0)
            if inputs[col] == 'Sélectionner...':
                inputs[col] = None
            elif inputs[col] == 'Autre':
                inputs[col] = ''
    
    # Submit button
    st.markdown("---")
    submitted = st.form_submit_button('🚀 Estimer le prix', use_container_width=True)

# Process form submission
if submitted:
    # Replace empty strings with None for categorical features
    # This ensures proper handling of missing values in the API
    payload = {k: (None if (isinstance(v, str) and v.strip() == '') else v) for k, v in inputs.items()}
    
    # Show loading indicator
    with st.spinner('⏳ Calcul du prix en cours...'):
        try:
            # Send prediction request to the FastAPI backend (app/api/main.py)
            # The backend's /predict endpoint processes this request and returns the predicted price
            resp = requests.post(f'{API_URL}/predict', json=payload, timeout=30)
            if resp.status_code == 200:
                # Extract and display the predicted price
                price = resp.json().get('price')
                st.success('')
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                    <h2 style='color: #1f77b4; margin-bottom: 10px;'>💰 Prix estimé</h2>
                    <h1 style='color: #2ecc71; font-size: 2.5em; margin: 0;'>{price:,.2f} €</h1>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Display error if API returns non-200 status
                st.error(f'❌ Erreur {resp.status_code}: {resp.text}')
        except Exception as e:
            # Handle connection errors or other exceptions
            st.error(f'❌ Impossible de contacter l\'API: {e}')
            st.info('💡 Assurez-vous que le backend API est en cours d\'exécution.')

# Display model feature information in an expandable section (optional)
st.markdown("---")
with st.expander('ℹ️ Informations techniques du modèle'):
    if expected_cols:
        # Create a simple list view
        st.write(f"**Nombre total de caractéristiques:** {len(expected_cols)}")
        st.write(f"**Caractéristiques numériques:** {len(numeric_cols)}")
        st.write(f"**Caractéristiques catégorielles:** {len(categorical_cols)}")
        if metadata.get('metrics'):
            metrics = metadata['metrics']
            st.write(f"**Précision du modèle (R²):** {metrics.get('r2', 0):.2%}")
    else:
        # Show message if metadata is not available
        st.warning("⚠️ Entraînez le modèle pour générer 'models/metadata.json'.")


