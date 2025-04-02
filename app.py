import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from datetime import datetime, timedelta, date
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("api_compatible_model.keras")
scaler = joblib.load("api_scaler.save")

# --- Constants ---
API_KEY = "4dfe8c91ad62e06e616bfb5599836562"
LAT, LON = 55.8642, -4.2518
UNITS = "metric"
TURBINE_CAPACITY_KW = 1.5
BATTERY_CAPACITY_KWH = 20.4
DAILY_USAGE_KWH = 12
IMPORT_RATE = 0.2582
EXPORT_RATE = 0.15
STANDING_CHARGE = 0.5513  # Daily standing charge in £

# --- Simulated Model ---
def predict_power(X_scaled):
    return model.predict(X_scaled, verbose=0).flatten()

# --- Feature Engineering Helper ---
def add_features(df):
    df['Humidity_Temp'] = df['RelHum_2m'] * df['Temp_2m']
    df['WindSpeed_Squared'] = df['WindSpeed'] ** 2
    df['WindSpeed_Cubic'] = df['WindSpeed'] ** 3
    df['Temp_Squared'] = df['Temp_2m'] ** 2
    df['Temp_Cubic'] = df['Temp_2m'] ** 3
    df['Humidity_Squared'] = df['RelHum_2m'] ** 2
    df['Humidity_Cubic'] = df['RelHum_2m'] ** 3
    df["Wind_Temp"] = df["WindSpeed"] * df["Temp_2m"]
    df["Wind_Humidity"] = df["WindSpeed"] * df["RelHum_2m"]
    return df

def fetch_hourly_weather(limit=96):
    url = f"https://pro.openweathermap.org/data/2.5/forecast/hourly?lat={LAT}&lon={LON}&appid={API_KEY}&units={UNITS}"
    response = requests.get(url).json()

    if response.get("cod") != "200":
        st.error(response.get("message", "Unknown error"))
        return None

    data = []
    for entry in response['list'][:limit]:
        dt = datetime.fromtimestamp(entry['dt'])
        data.append({
            'datetime': dt,
            'Temp_2m': entry['main']['temp'],
            'RelHum_2m': entry['main']['humidity'],
            'WindSpeed': entry['wind']['speed']
        })

    df = pd.DataFrame(data)
    return df

def get_current_weather():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units={UNITS}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    return {
        "🌡️ Temperature (°C)": data["main"]["temp"],
        "💧 Humidity (%)": data["main"]["humidity"],
        "🌬️ Wind Speed (m/s)": data["wind"]["speed"],
        "☁️ Cloud Coverage (%)": data["clouds"]["all"],
        "🌤️ Condition": data["weather"][0]["description"].title()
    }

# --- Streamlit App ---
# --- Streamlit Config ---
st.set_page_config(page_title="Wind Turbine Energy Simulator", layout="wide")

# --- Option Menu ---
selected_option = st.sidebar.selectbox(
    "🌤️ Choose Simulation Option",
    [
        "📊 Select a simulation mode...",
        "1. 24-Hour Prediction ",
        "2. Bedtime Battery Prediction  ",
        "3. Multi-day Forecast ",
        "4. 4-Day Hourly Forecast",
        "5. Holiday Mode 🏝️",
    ]
)

# --- Home screen (only shown if default option is selected) ---
if selected_option == "📊 Select a simulation mode...":
    st.markdown("""
    # 🌬️ Wind Turbine Energy Prediction ⚡

    Welcome to the Wind Turbine power prediction app! This tool helps estimate how much energy your turbine could generate based on weather forecasts and how it interacts with your home's battery, usage, and export income.

    ### 📋 **Features:**
    - 🔹 24-hour prediction based on real-time weather  
    - 🌙 Bedtime battery analysis  
    - 📆 Multi-day forecasts  
    - 🏖️ Holiday mode  

    Use the sidebar to get started with a prediction mode.
    """)

    st.divider()

    st.subheader("⚙️ Battery Configuration")
    with st.expander("🔋 Adjust Battery Settings", expanded=True):
        if "battery_kwh" not in st.session_state:
            st.session_state.battery_kwh = 0.5 * BATTERY_CAPACITY_KWH  # Default 50%

        reset_battery = st.checkbox("🔁 Reset battery to slider value", value=False)

        battery_percent = st.slider(
            "🔋 Select current battery percentage",
            0, 100,
            int((st.session_state.battery_kwh / BATTERY_CAPACITY_KWH) * 100)
        )

        if reset_battery:
            st.session_state.battery_kwh = (battery_percent / 100) * BATTERY_CAPACITY_KWH
    st.divider()
    st.subheader("🌦️ Current Weather Conditions")
    weather = get_current_weather()
    if weather:
        for label, value in weather.items():
            st.markdown(f"**{label}:** {value}")
    else:
        st.warning("Unable to fetch current weather data. Please try again later.")


# --- Show welcome screen only if no option selected ---
if selected_option == "📊 Select a simulation mode...":
    st.markdown("### 👈 Use the menu on the left to begin a simulation.")


# --- Option 1: 24-Hour Prediction ---
elif selected_option.startswith("1"):
    st.subheader("🔹 24-Hour Energy Forecast")
    url = f"https://pro.openweathermap.org/data/2.5/forecast/hourly?lat={LAT}&lon={LON}&appid={API_KEY}&units={UNITS}"
    response = requests.get(url).json()

    if response.get("cod") != "200":
        st.error(response.get("message", "Unknown error"))
    else:
        data = []
        for entry in response['list'][:24]:
            dt = datetime.fromtimestamp(entry['dt'])
            data.append({
                'datetime': dt,
                'Temp_2m': entry['main']['temp'],
                'RelHum_2m': entry['main']['humidity'],
                'WindSpeed': entry['wind']['speed']
            })
        df = pd.DataFrame(data)
        df = add_features(df)
        features = [
            'Temp_2m', 'RelHum_2m', 'WindSpeed', 'Humidity_Temp',
            'WindSpeed_Squared', 'WindSpeed_Cubic',
            'Temp_Squared', 'Temp_Cubic',
            'Humidity_Squared', 'Humidity_Cubic',
            'Wind_Temp', 'Wind_Humidity'
        ]

        X_scaled = scaler.transform(df[features])

        df['PredictedPower'] = predict_power(X_scaled)
        df['Energy_kWh'] = df['PredictedPower'] * TURBINE_CAPACITY_KW

        total_energy = df['Energy_kWh'].sum()
        co2_saved = total_energy * 0.147
        battery_savings = min(total_energy, DAILY_USAGE_KWH)
        battery_saving_cost = battery_savings * IMPORT_RATE
        net_energy = total_energy - DAILY_USAGE_KWH

        if net_energy >= 0:
            export_income = net_energy * EXPORT_RATE
            import_cost = 0
        else:
            export_income = 0
            import_cost = abs(net_energy) * IMPORT_RATE

        total_cost = STANDING_CHARGE + import_cost - export_income

        battery_kwh = st.session_state.battery_kwh + net_energy
        export_kwh = max(battery_kwh - BATTERY_CAPACITY_KWH, 0)
        battery_kwh = min(max(battery_kwh, 0), BATTERY_CAPACITY_KWH)
        st.session_state.battery_kwh = battery_kwh

        st.metric("🔋 Updated Battery Level", f"{(battery_kwh / BATTERY_CAPACITY_KWH) * 100:.2f}%")

        st.write(f"**🔹 Predicted Energy Generated:** {total_energy:.2f} kWh")
        st.write(f"**🌍 CO₂ Savings:** {co2_saved:.2f} kg")
        st.write(f"**🔹 Household Usage:** {DAILY_USAGE_KWH:.2f} kWh")
        st.write(f"**🔹 Net Surplus:** {net_energy:.2f} kWh")
        if export_income > 0:
            st.success(f"💷 Export Income: £{export_income:.2f}")
        else:
            st.warning(f"💷 Import Cost: £{import_cost:.2f}")
        st.info(f"✅ Battery Savings: £{battery_saving_cost:.2f}")
        st.markdown(f"💰 **Total Cost for Day:** £{total_cost:.2f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['datetime'], df['Energy_kWh'], marker='o')
        ax.set_title('24-Hour Forecast')
        ax.set_ylabel('Energy (kWh)')
        ax.set_xlabel('Time')
        ax.grid(True)
        fig.autofmt_xdate()
        st.pyplot(fig)

# --- OPTION 2: Bedtime Battery Prediction ---
elif selected_option.startswith("2"):
    st.subheader("🛌 Bedtime Battery Prediction")
    bedtime = st.slider("Select bedtime hour", 0, 23, 23)
    wake = st.slider("Select wake-up hour", 0, 23, 7)

    url = f"https://pro.openweathermap.org/data/2.5/forecast/hourly?lat={LAT}&lon={LON}&appid={API_KEY}&units={UNITS}"
    response = requests.get(url).json()

    if response.get("cod") != "200":
        st.error(response.get("message", "Unknown error"))
    else:
        data = []
        for entry in response['list'][:24]:
            dt = datetime.fromtimestamp(entry['dt'])
            data.append({
                'datetime': dt,
                'hour': dt.hour,
                'Temp_2m': entry['main']['temp'],
                'RelHum_2m': entry['main']['humidity'],
                'WindSpeed': entry['wind']['speed']
            })
        df = pd.DataFrame(data)
        df = add_features(df)

        features = [
            'Temp_2m', 'RelHum_2m', 'WindSpeed', 'Humidity_Temp',
            'WindSpeed_Squared', 'WindSpeed_Cubic',
            'Temp_Squared', 'Temp_Cubic',
            'Humidity_Squared', 'Humidity_Cubic',
            'Wind_Temp', 'Wind_Humidity'
        ]

        X_scaled = scaler.transform(df[features])

        df['PredictedPower'] = predict_power(X_scaled)
        df['Energy_kWh'] = df['PredictedPower'] * TURBINE_CAPACITY_KW

        if wake > bedtime:
            period = df[(df['hour'] >= bedtime) & (df['hour'] <= wake)]
        else:
            period = df[(df['hour'] >= bedtime) | (df['hour'] <= wake)]

        energy_kwh = period['Energy_kWh'].sum()
        co2_saved = energy_kwh * 0.147
        fridge_usage = (1 / 24) * len(period)
        net_energy = energy_kwh - fridge_usage

        import_cost = max(-net_energy, 0) * IMPORT_RATE
        export_income = max(net_energy, 0) * EXPORT_RATE
        battery_saving_cost = min(energy_kwh, DAILY_USAGE_KWH) * IMPORT_RATE
        total_cost = STANDING_CHARGE + import_cost - export_income

        new_battery_kwh = st.session_state.battery_kwh + net_energy
        export_kwh = max(new_battery_kwh - BATTERY_CAPACITY_KWH, 0)
        new_battery_kwh = min(max(new_battery_kwh, 0), BATTERY_CAPACITY_KWH)
        st.session_state.battery_kwh = new_battery_kwh

        st.metric("🔋 Updated Battery Level", f"{(new_battery_kwh / BATTERY_CAPACITY_KWH) * 100:.2f}%")

        st.write(f"🔹 **Predicted Energy Generated:** {energy_kwh:.2f} kWh")
        st.write(f"🌍 **CO₂ Savings:** {co2_saved:.2f} kg")
        st.write(f"🔌 **Fridge Usage:** {fridge_usage:.2f} kWh")
        st.write(f"⚡ **Net Surplus:** {net_energy:.2f} kWh")
        st.write(f"💷 **Export Income:** £{export_income:.2f}")
        st.write(f"💷 **Import Cost:** £{import_cost:.2f}")
        st.write(f"✅ **Battery Savings:** £{battery_saving_cost:.2f}")
        st.write(f"💰 **Total Cost:** £{total_cost:.2f}")

        # Order period chronologically across midnight if needed
        period = period.sort_values(by='datetime')

        # Use datetime on x-axis for correct ordering
        st.line_chart(period.set_index('datetime')['Energy_kWh'])


# --- OPTION 3: MULTI-DAY DAILY API ---
elif selected_option.startswith("3"):
    st.subheader("📆 Multi-day Forecast")
    days = st.slider("Select number of days to forecast", 1, 16, 7)
    url = f"https://api.openweathermap.org/data/2.5/forecast/daily?lat={LAT}&lon={LON}&cnt={days}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()

    if response.get("cod") not in ("200", 200):
        st.error(f"Error fetching data: {response.get('message', 'Unknown error')}")
    else:
        data = []
        for entry in response['list']:
            dt = datetime.fromtimestamp(entry['dt'])
            data.append({
                'date': dt.date(),
                'Temp_2m': entry['temp']['day'],
                'RelHum_2m': entry['humidity'],
                'WindSpeed': entry['speed']
            })
        df = pd.DataFrame(data)
        df = add_features(df)
        features = [
            'Temp_2m', 'RelHum_2m', 'WindSpeed', 'Humidity_Temp',
            'WindSpeed_Squared', 'WindSpeed_Cubic',
            'Temp_Squared', 'Temp_Cubic',
            'Humidity_Squared', 'Humidity_Cubic',
            'Wind_Temp', 'Wind_Humidity'
        ]

        X_scaled = scaler.transform(df[features])
        df['PredictedPower'] = predict_power(X_scaled)
        df['Energy_kWh'] = df['PredictedPower'] * TURBINE_CAPACITY_KW * 24
        df['CO2_Saved'] = df['Energy_kWh'] * 0.147
        df['Battery_Saving_kWh'] = np.minimum(df['Energy_kWh'], DAILY_USAGE_KWH)
        df['Battery_Saving_Value'] = df['Battery_Saving_kWh'] * IMPORT_RATE
        df['Net_Surplus'] = df['Energy_kWh'] - DAILY_USAGE_KWH
        df['Export_Income'] = np.where(df['Net_Surplus'] > 0, df['Net_Surplus'] * EXPORT_RATE, 0)
        df['Import_Cost'] = np.where(df['Net_Surplus'] < 0, abs(df['Net_Surplus']) * IMPORT_RATE, 0)
        df['Net_Cost'] = STANDING_CHARGE + df['Import_Cost'] - df['Export_Income']

        st.dataframe(df[['date', 'Energy_kWh', 'CO2_Saved', 'Net_Surplus', 'Export_Income', 'Import_Cost', 'Net_Cost']])
        st.bar_chart(df.set_index('date')[['Energy_kWh']])

# --- OPTION 4: 4-DAY HOURLY FORECAST ---
elif selected_option.startswith("4"):
    st.subheader("⏱️ 4-Day Hourly Forecast")
    df = fetch_hourly_weather(96)
    if df is not None:
        df = add_features(df)
        features = [
            'Temp_2m', 'RelHum_2m', 'WindSpeed', 'Humidity_Temp',
            'WindSpeed_Squared', 'WindSpeed_Cubic',
            'Temp_Squared', 'Temp_Cubic',
            'Humidity_Squared', 'Humidity_Cubic',
            'Wind_Temp', 'Wind_Humidity'
        ]

        X_scaled = scaler.transform(df[features])
        df['PredictedPower'] = predict_power(X_scaled)
        df['Energy_kWh'] = df['PredictedPower'] * TURBINE_CAPACITY_KW
        df['Date'] = df['datetime'].dt.date

        daily_summary = df.groupby('Date')['Energy_kWh'].sum().reset_index()
        daily_summary['Net_Surplus'] = daily_summary['Energy_kWh'] - DAILY_USAGE_KWH
        daily_summary['Export_Income'] = np.where(daily_summary['Net_Surplus'] > 0, daily_summary['Net_Surplus'] * EXPORT_RATE, 0)
        daily_summary['Import_Cost'] = np.where(daily_summary['Net_Surplus'] < 0, abs(daily_summary['Net_Surplus']) * IMPORT_RATE, 0)
        daily_summary['Net_Cost'] = STANDING_CHARGE + daily_summary['Import_Cost'] - daily_summary['Export_Income']

        st.dataframe(daily_summary)
        st.line_chart(df.set_index('datetime')['Energy_kWh'])

# --- OPTION 5: Holiday Mode ---
elif selected_option.startswith("5"):
    st.subheader("🏝️ Holiday Mode")
    today = date.today()
    return_date = st.date_input("Select return date:", min_value=today + timedelta(days=1), max_value=today + timedelta(days=16))
    days_away = (return_date - today).days

    url = f"https://api.openweathermap.org/data/2.5/forecast/daily?lat={LAT}&lon={LON}&cnt={days_away}&appid={API_KEY}&units={UNITS}"
    response = requests.get(url).json()

    if response.get("cod") not in ("200", 200):
        st.error(response.get("message", "Unknown error"))
    else:
        data = []
        for entry in response['list']:
            dt = datetime.fromtimestamp(entry['dt']).date()
            data.append({
                'date': dt,
                'Temp_2m': entry['temp']['day'],
                'RelHum_2m': entry['humidity'],
                'WindSpeed': entry['speed']
            })
        df = pd.DataFrame(data)
        df = add_features(df)

        features = [
            'Temp_2m', 'RelHum_2m', 'WindSpeed', 'Humidity_Temp',
            'WindSpeed_Squared', 'WindSpeed_Cubic',
            'Temp_Squared', 'Temp_Cubic',
            'Humidity_Squared', 'Humidity_Cubic',
            'Wind_Temp', 'Wind_Humidity'
        ]

        X_scaled = scaler.transform(df[features])

        df['PredictedPower'] = predict_power(X_scaled)
        df['Energy_kWh'] = df['PredictedPower'] * TURBINE_CAPACITY_KW * 24
        df['Net_Surplus'] = df['Energy_kWh'] - 1.0

        battery_kwh = st.session_state.battery_kwh
        summaries = []

        for _, row in df.iterrows():
            free_export = min(battery_kwh, 13.2)
            battery_kwh -= free_export
            free_charge = min(15, BATTERY_CAPACITY_KWH - battery_kwh)
            battery_kwh += free_charge
            battery_kwh += row['Net_Surplus']

            if battery_kwh > BATTERY_CAPACITY_KWH:
                export_kwh = battery_kwh - BATTERY_CAPACITY_KWH
                battery_kwh = BATTERY_CAPACITY_KWH
            else:
                export_kwh = 0

            if battery_kwh < 0:
                import_kwh = abs(battery_kwh)
                battery_kwh = 0
            else:
                import_kwh = 0

            export_income = (export_kwh + free_export) * EXPORT_RATE
            import_cost = import_kwh * IMPORT_RATE
            battery_offset = min(row['Energy_kWh'], 1.0) * IMPORT_RATE
            co2_saved = row['Energy_kWh'] * 0.147
            net_cost = STANDING_CHARGE + import_cost - export_income

            summaries.append({
                'Date': row['date'],
                'PredictedEnergy': row['Energy_kWh'],
                'CO2Saved': co2_saved,
                'FreeExport': free_export,
                'FreeCharge': free_charge,
                'Exported': export_kwh,
                'Imported': import_kwh,
                'BatteryLevel': battery_kwh,
                'ExportIncome': export_income,
                'ImportCost': import_cost,
                'BatteryOffset': battery_offset,
                'NetCost': net_cost
            })

        st.session_state.battery_kwh = battery_kwh
        summary_df = pd.DataFrame(summaries)
        st.dataframe(summary_df)
        st.bar_chart(summary_df.set_index('Date')['PredictedEnergy'])
        
        # --- Holiday Summary Stats ---
        total_energy = summary_df['PredictedEnergy'].sum()
        total_co2 = summary_df['CO2Saved'].sum()
        total_offset = summary_df['BatteryOffset'].sum()
        total_import = summary_df['Imported'].sum()
        total_export = summary_df['Exported'].sum()
        total_free_export = summary_df['FreeExport'].sum()
        total_free_charge = summary_df['FreeCharge'].sum()
        total_income = summary_df['ExportIncome'].sum()
        total_import_cost = summary_df['ImportCost'].sum()
        total_net_cost = summary_df['NetCost'].sum()
        final_battery = summary_df['BatteryLevel'].iloc[-1]

        # --- Display Summary ---
        st.markdown("### 📋 Holiday Summary")
        st.write(f"🔹 **Predicted Energy:** {total_energy:.2f} kWh")
        st.write(f"🌍 **Predicted CO₂ Savings:** {total_co2:.2f} kg")
        st.write(f"🔹 **Predicted Holiday Usage:** {1.0 * len(summary_df):.2f} kWh")
        st.write(f"🔹 **Predicted Net Surplus:** {total_energy - (1.0 * len(summary_df)):.2f} kWh")
        st.write(f"🔋 **Battery Level:** {final_battery / BATTERY_CAPACITY_KWH * 100:.2f}% ({final_battery:.2f} kWh)")
        st.write(f"⏱️ **Predicted Export (00:01–02:00):** {total_free_export:.2f} kWh")
        st.write(f"⏱️ **Predicted Free Charge (02:00–04:00):** {total_free_charge:.2f} kWh")
        st.write(f"⚡ **Predicted Exported:** {total_export:.2f} kWh → £{total_income:.2f}")
        st.write(f"✅ **Battery Offset (Saved from Grid):** £{total_offset:.2f}")
        st.write(f"💷 **Standing Charge:** £{STANDING_CHARGE * len(summary_df):.2f}")
        st.markdown(f"💰 **Net Cost:** £{total_net_cost:.2f}")
