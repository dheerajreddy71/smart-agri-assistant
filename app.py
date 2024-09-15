import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from googletrans import Translator

# Load and prepare datasets for yield prediction
yield_df = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/yield_df.csv")
crop_recommendation_data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/Crop_recommendation.csv")

yield_preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0, 1, 2, 3]),
        ('OHE', OneHotEncoder(drop='first'), [4, 5]),
    ],
    remainder='passthrough'
)
yield_X = yield_df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']]
yield_y = yield_df['hg/ha_yield']
yield_X_train, yield_X_test, yield_y_train, yield_y_test = train_test_split(yield_X, yield_y, train_size=0.8, random_state=0, shuffle=True)
yield_X_train_dummy = yield_preprocessor.fit_transform(yield_X_train)
yield_X_test_dummy = yield_preprocessor.transform(yield_X_test)
yield_model = KNeighborsRegressor(n_neighbors=5)
yield_model.fit(yield_X_train_dummy, yield_y_train)

crop_X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
crop_y = crop_recommendation_data['label']
crop_X_train, crop_X_test, crop_y_train, crop_y_test = train_test_split(crop_X, crop_y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(crop_X_train, crop_y_train)

# Load crop data and train the model for temperature prediction
data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/ds1.csv", encoding='ISO-8859-1')
data = data.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
X = data.drop(['Crop', 'Temperature Required (°F)'], axis=1)
y = data['Temperature Required (°F)']
model = LinearRegression()
model.fit(X, y)

# Function to predict temperature and humidity requirements for a crop
def predict_requirements(crop_name):
    crop_name = crop_name.lower()
    crop_data = data[data['Crop'].str.lower() == crop_name].drop(['Crop', 'Temperature Required (°F)'], axis=1)
    if crop_data.empty:
        return None, None  # Handle cases where crop_name is not found
    predicted_temperature = model.predict(crop_data)
    crop_row = data[data['Crop'].str.lower() == crop_name]
    humidity_required = crop_row['Humidity Required (%)'].values[0]
    return humidity_required, predicted_temperature[0]

# Function to get pest warnings for a crop
crop_pest_data = {}
planting_time_info = {}
growth_stage_info = {}
pesticides_info = {}

# Read data from the CSV file and store it in dictionaries
pest_data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/raw/main/ds2.csv")
for _, row in pest_data.iterrows():
    crop = row[0].strip().lower()
    pest = row[1].strip()
    crop_pest_data[crop] = pest
    planting_time_info[crop] = row[5].strip()
    growth_stage_info[crop] = row[6].strip()
    pesticides_info[crop] = row[4].strip()

def predict_pest_warnings(crop_name):
    crop_name = crop_name.lower()
    specified_crops = [crop_name]

    pest_warnings = []

    for crop in specified_crops:
        if crop in crop_pest_data:
            pests = crop_pest_data[crop].split(', ')
            warning_message = f"\nBeware of pests like {', '.join(pests)} for {crop.capitalize()}.\n"

            if crop in planting_time_info:
                planting_time = planting_time_info[crop]
                warning_message += f"\nPlanting Time: {planting_time}\n"

            if crop in growth_stage_info:
                growth_stage = growth_stage_info[crop]
                warning_message += f"\nGrowth Stages of Plant: {growth_stage}\n"

            if crop in pesticides_info:
                pesticides = pesticides_info[crop]
                warning_message += f"\nUse Pesticides like: {pesticides}\n"
                
            pest_warnings.append(warning_message)

    return '\n'.join(pest_warnings)

# Load and preprocess crop price data
price_data = pd.read_csv('https://github.com/dheerajreddy71/Design_Project/raw/main/pred_data.csv', encoding='ISO-8859-1')
price_data['arrival_date'] = pd.to_datetime(price_data['arrival_date'])
price_data['day'] = price_data['arrival_date'].dt.day
price_data['month'] = price_data['arrival_date'].dt.month
price_data['year'] = price_data['arrival_date'].dt.year
price_data.drop(['arrival_date'], axis=1, inplace=True)

price_X = price_data.drop(['min_price', 'max_price', 'modal_price'], axis=1)
price_y = price_data[['min_price', 'max_price', 'modal_price']]

price_encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['state', 'district', 'market', 'commodity', 'variety'])
    ],
    remainder='passthrough'
)

price_X_encoded = price_encoder.fit_transform(price_X)
price_X_train, price_X_test, price_y_train, price_y_test = train_test_split(price_X_encoded, price_y, test_size=0.2, random_state=42)

price_model = LinearRegression()
price_model.fit(price_X_train, price_y_train)

# Streamlit app
st.set_page_config(page_title="Smart Agri Assistant", layout="wide", page_icon="🌾")

# Add a background image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://github.com/dheerajreddy71/Webbuild/raw/main/background.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Smart Agri Assistant")




# Yield Prediction
st.header("Predict Crop Yield")
year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
rainfall = st.number_input("Average Rainfall (mm per year)")
pesticides = st.number_input("Pesticides Used (tonnes)")
temp = st.number_input("Average Temperature (°C)")
area = st.text_input("Area(Country)")
item = st.text_input("Item(Crop Name)")

if st.button("Predict Yield"):
    features = {
        'Year': year,
        'average_rain_fall_mm_per_year': rainfall,
        'pesticides_tonnes': pesticides,
        'avg_temp': temp,
        'Area': area,
        'Item': item,
    }
    features_array = np.array([[features['Year'], features['average_rain_fall_mm_per_year'],
                                features['pesticides_tonnes'], features['avg_temp'],
                                features['Area'], features['Item']]], dtype=object)
    transformed_features = yield_preprocessor.transform(features_array)
    predicted_yield = yield_model.predict(transformed_features).reshape(1, -1)
    st.success(f"The predicted yield is {predicted_yield[0][0]:.2f} hectograms (hg) per hectare (ha).")

# Crop Recommendation
st.header("Recommend Crops")
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("Recommend Crop"):
    crop_features = [N, P, K, temperature, humidity, ph, rainfall]
    recommended_crop = crop_model.predict([crop_features])[0]
    st.success(f"Recommended Crop: {recommended_crop}")

# Crop Requirements and Pest Warnings
st.header("Predict Crop Requirements and Pest Warnings")
crop_name = st.text_input("Crop Name")

if st.button("Calculate"):
    humidity, temperature = predict_requirements(crop_name)
    pest_warning = predict_pest_warnings(crop_name)
    if humidity is not None and temperature is not None:
        st.info(f"Humidity Required: {humidity}%")
        st.info(f"Temperature Required: {temperature:.2f}°F")
    else:
        st.warning("Crop not found.")
    st.info(f"Pest Warnings: {pest_warning}")

# Crop Price Prediction
st.header("Predict Crop Prices")
state = st.text_input("State")
district = st.text_input("District")
market = st.text_input("Market")
commodity = st.text_input("Commodity")
variety = st.text_input("Variety")
arrival_date = st.date_input("Arrival Date")

if st.button("Predict Prices"):
    if not all([state, district, market, commodity, variety, arrival_date]):
        st.error("Please provide all inputs.")
    else:
        input_data = {
            'state': state,
            'district': district,
            'market': market,
            'commodity': commodity,
            'variety': variety,
            'arrival_date': pd.to_datetime(arrival_date)
        }

        input_df = pd.DataFrame([input_data])
        input_df['day'] = input_df['arrival_date'].dt.day
        input_df['month'] = input_df['arrival_date'].dt.month
        input_df['year'] = input_df['arrival_date'].dt.year
        input_df.drop(['arrival_date'], axis=1, inplace=True)
        input_encoded = price_encoder.transform(input_df)

        predicted_prices = price_model.predict(input_encoded)
        min_price, max_price, modal_price = predicted_prices[0]
        st.success(f"Predicted Prices - Min: {min_price}, Max: {max_price}, Modal: {modal_price}")

# API keys
geocoding_api_key = '80843f03ed6b4945a45f1bd8c51e5c2f'
weather_api_key = 'b53305cd6b960c1984aed0acaf76aa2e'

# Translator setup
translator = Translator()

# Function to translate text based on selected language
def translate_text(text, dest_lang):
    try:
        translation = translator.translate(text, dest=dest_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Function to get geolocation
def get_lat_lon(village_name):
    geocoding_url = f'https://api.opencagedata.com/geocode/v1/json?q={village_name}&key={geocoding_api_key}'
    try:
        response = requests.get(geocoding_url)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            latitude = data['results'][0]['geometry']['lat']
            longitude = data['results'][0]['geometry']['lng']
            return latitude, longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Error fetching geocoding data: {e}")
        return None, None

# Function to get weather forecast
def get_weather_forecast(latitude, longitude, dest_lang):
    weather_url = f'https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&units=metric&cnt=40&appid={weather_api_key}'
    try:
        response = requests.get(weather_url)
        response.raise_for_status()
        data = response.json()
        if data['cod'] == '200':
            forecast = []
            for item in data['list']:
                date_time = item['dt_txt']
                date, time = date_time.split(' ')
                weather_desc = translate_text(item['weather'][0]['description'], dest_lang)
                forecast.append({
                    'date': date,
                    'time': time,
                    'temp': item['main']['temp'],
                    'pressure': item['main']['pressure'],
                    'humidity': item['main']['humidity'],
                    'weather': weather_desc
                })
            return forecast
        else:
            st.error(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# Load and preprocess data for fertilizer recommendation
url = 'https://raw.githubusercontent.com/dheerajreddy71/Design_Project/main/fertilizer_recommendation.csv'
data = pd.read_csv(url)
data.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'}, inplace=True)
data.dropna(inplace=True)

# Encode categorical variables
encode_soil = LabelEncoder()
data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)

encode_crop = LabelEncoder()
data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)

encode_ferti = LabelEncoder()
data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data.drop('Fertilizer', axis=1), data.Fertilizer, test_size=0.2, random_state=1)

# Train a Random Forest Classifier
rand = RandomForestClassifier()
rand.fit(x_train, y_train)

# Streamlit app
st.set_page_config(page_title="Smart Agri Assistant", layout="wide", page_icon="🌾")

# Add a background image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://github.com/dheerajreddy71/Webbuild/raw/main/background.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Language selection
st.sidebar.header("Language Selection")
languages = {'English': 'en', 'Hindi': 'hi', 'Tamil': 'ta', 'Telugu': 'te'}
selected_language = st.sidebar.selectbox('Select language', list(languages.keys()))

# Get the selected language code
dest_lang = languages[selected_language]

# Weather Forecast Section
st.header(translate_text("Weather Forecast for Village", dest_lang))
village_name = st.text_input(translate_text('Enter village name', dest_lang), key='village_name_input')

if st.button(translate_text('Fetch Weather', dest_lang), key='fetch_weather_button'):
    if village_name:
        latitude, longitude = get_lat_lon(village_name)
        if latitude and longitude:
            st.write(translate_text(f'Coordinates: Latitude {latitude}, Longitude {longitude}', dest_lang))
            forecast = get_weather_forecast(latitude, longitude, dest_lang)
            if forecast:
                df = pd.DataFrame(forecast)
                st.write(translate_text('Weather Forecast:', dest_lang))
                st.dataframe(df)
            else:
                st.write(translate_text('Weather forecast data not available.', dest_lang))
        else:
            st.write(translate_text('Village not found.', dest_lang))
    else:
        st.write(translate_text('Please enter a village name.', dest_lang))

# Fertilizer Recommendation Section
st.header(translate_text("Fertilizer Recommendation System", dest_lang))

# Input fields for fertilizer recommendation
temperature = st.number_input(translate_text('Temperature', dest_lang), format="%.2f", key='temperature_input')
humidity = st.number_input(translate_text('Humidity', dest_lang), format="%.2f", key='humidity_input')
moisture = st.number_input(translate_text('Moisture', dest_lang), format="%.2f", key='moisture_input')
soil_type = st.selectbox(translate_text('Soil Type', dest_lang), encode_soil.classes_, key='soil_type_input')
crop_type = st.selectbox(translate_text('Crop Type', dest_lang), encode_crop.classes_, key='crop_type_input')
nitrogen = st.number_input(translate_text('Nitrogen', dest_lang), format="%.2f", key='nitrogen_input')
potassium = st.number_input(translate_text('Potassium', dest_lang), format="%.2f", key='potassium_input')
phosphorous = st.number_input(translate_text('Phosphorous', dest_lang), format="%.2f", key='phosphorous_input')

if st.button(translate_text('Predict Fertilizer', dest_lang), key='predict_fertilizer_button'):
    try:
        soil_type_encoded = encode_soil.transform([soil_type])[0]
        crop_type_encoded = encode_crop.transform([crop_type])[0]
        prediction = rand.predict([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]])
        recommended_fertilizer = encode_ferti.inverse_transform(prediction)[0]
        st.write(translate_text(f"Recommended Fertilizer: {recommended_fertilizer}", dest_lang))
    except Exception as e:
        st.error(translate_text(f"Error: {e}", dest_lang))
