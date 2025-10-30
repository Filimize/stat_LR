import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# --- Load model coefficients ---
coefs = pd.read_csv("model_coeffs.csv")
means = pd.read_csv("center_values.csv")
# Extract intercept
intercept = coefs.loc[coefs['term'] == '(Intercept)', 'estimate'].values[0]
coefs = coefs[coefs['term'] != '(Intercept)']

coef_dict = dict(zip(coefs['term'], coefs['estimate']))

# --- Centering values (from R means) ---
AVG_TEMP_MEAN = means.loc[means['variable'] == 'Avg_Temp', 'mean_value'].values[0]
RAIN_MEAN = means.loc[means['variable'] == 'Rain', 'mean_value'].values[0]

# --- Streamlit UI ---
# --- User chooses theme
st.image("head_LR.png", use_container_width=True)
st.title("PM2.5 Prediction \n### (Linear Regression)")

#-- model description ---
st.markdown(
    """
    This application predicts PM2.5 concentration based on environmental factors using a linear regression model.
    The model has adjusted R² of approximately 74.93%.
    Please input the required parameters below to get the prediction along with confidence intervals.
    """
)
# --- User Inputs ---
PM25_lag1 = st.number_input("PM25_lag1", min_value=0.0, max_value=500.0, value=50.0)
Avg_Temp = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=27.0)
Rain = st.number_input("Rain", min_value=0.0, max_value=200.0, value=5.0)
Re_Humid = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
Wind = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=2.0)
Sun = st.number_input("Sun", min_value=0.0, max_value=24.0, value=6.0)

# --- Center raw inputs ---
Avg_Temp_c = Avg_Temp - AVG_TEMP_MEAN
Rain_c = Rain - RAIN_MEAN

# --- Predict ---
PM25_pred = (
    intercept
    + coef_dict.get("PM25_lag1", 0) * PM25_lag1
    + coef_dict.get("Avg_Temp_c", 0) * Avg_Temp_c
    + coef_dict.get("Rain_c", 0) * Rain_c
    + coef_dict.get("Re_Humid", 0) * Re_Humid
    + coef_dict.get("Wind", 0) * Wind
    + coef_dict.get("Sun", 0) * Sun
    + coef_dict.get("Avg_Temp_c:Rain_c", 0) * Avg_Temp_c * Rain_c
)
PM25_pred = max(PM25_pred, 0) # Ensure non-negative prediction
confidence = st.selectbox("Select Confidence Level", [90, 95, 99])
z = {90: 1.64, 95: 1.96, 99: 2.58}[confidence]
# --- Assume fixed std error (you can import from R summary) ---
std_error = 23.65
lower = max(0, PM25_pred - z * std_error)
upper = min(500, PM25_pred + z * std_error)

# --- Visualization ---

colors = ["#7FFF00", "#FFFF00", "#FFA500", "#FF4500", "#FF0000", "#800000"]
positions = [0, 0.2, 0.4, 0.6, 0.8, 1]
cmap = LinearSegmentedColormap.from_list("aqi_gradient", list(zip(positions, colors)))

fig, ax = plt.subplots(figsize=(8, 4))

# Generate stronger gradient background
x = np.linspace(0, 500, 2000)
y = np.linspace(0, 1, 2)
X, Y = np.meshgrid(x, y)
Z = X
ax.imshow(
    Z,
    aspect='auto',
    extent=[0, 500, ax.get_ylim()[0], ax.get_ylim()[1]],
    origin='lower',
    cmap=cmap,
    alpha=0.8
)

# Baseline
ax.plot([0, 500], [0.5, 0.5], color='black', linewidth=1.5)

# --- Confidence Interval Bar ---
ax.hlines(y=0.5, xmin=lower, xmax=upper, color='black', linewidth=3, alpha=0.9)
ax.plot(PM25_pred, 0.5, 'o', color='black', markersize=8)  # central point
ax.plot([lower, lower], [0.47, 0.53], color='black', linewidth=2)  # left cap
ax.plot([upper, upper], [0.47, 0.53], color='black', linewidth=2)  # right cap


ax.text(PM25_pred, 0.57, f"{PM25_pred:.1f} ", ha='center', fontsize=10, fontweight='bold', color='black')
ax.text(upper + 10, 0.57, f"{confidence}% CI", va='center', fontsize=9, color='black')

# Labels & formatting
ax.set_xlim(0, 500)
ax.set_xlabel("PM2.5 Concentration")
ax.set_title(f"PM2.5 = {round(PM25_pred,2)} (CI {confidence}%: {round(lower,2)} - {round(upper,2)})", fontsize=10, fontweight='bold')

# Styling
ax.grid(False)
ax.set_facecolor("white")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

st.pyplot(fig)
st.image("AQI.png", caption="Air Quality Index (AQI) Levels from https://aqicn.org/faq/2013-09-09/revised-pm25-aqi-breakpoints/", use_container_width=True)

