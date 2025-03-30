# Common Imports
import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import toml

import os
import requests
import json
import threading
import subprocess

# Preprocesing Imports
import pandas as pd
import time
from PIL import Image

import ollama
import altair as alt
from vega_datasets import data
from flask_api import ensure_markdown_files
# Warning
import warnings
warnings.filterwarnings('ignore')

import os


# Models
from Models_Management import WTV_Models
WTVM = WTV_Models()
model = WTVM.model
cities = WTVM.cities

from Vizualisations import TC_Pie_Plot, GlobeT_Chart, USAT_Chart
# Streamlit Config Functions
image_path = os.path.join(os.getcwd(), "Images", "globe.jpg")
image1 = Image.open(image_path)
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def predict_future(model, df, periods):
    future = df[["ds"]].copy()  # Ensure 'ds' column is properly copied
    last_date = future["ds"].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="D")[1:]
    future = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]



st.set_page_config(page_title='WTV', layout='wide')
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

def calculate_stats(df, level):
    highest_temp_record = df.loc[df['AverageTemperature'].idxmax()]
    lowest_temp_record = df.loc[df['AverageTemperature'].idxmin()]
    stats = {
        "highest_temp": {
            "country": highest_temp_record['Country'],
            "city": highest_temp_record['City'],
            "temperature": highest_temp_record['AverageTemperature'],
            "date": highest_temp_record['dt']
        },
        "lowest_temp": {
            "country": highest_temp_record['Country'],
            "city": highest_temp_record['City'],
            "temperature": lowest_temp_record['AverageTemperature'],
            "date": lowest_temp_record['dt']
        },
        "highest_avg_annual_temp": df.groupby([level, df['dt'].dt.year])['AverageTemperature'].mean().idxmax(),
        "lowest_avg_annual_temp": df.groupby([level, df['dt'].dt.year])['AverageTemperature'].mean().idxmin(),
        "countries_below_0_degrees_count": len(df[df['AverageTemperature'] < 0][level].unique()),
        "countries_above_35_degrees_count": len(df[df['AverageTemperature'] > 35][level].unique())
    }
    return stats


def get_stats(df):
    # Convertir la columna 'dt' a formato de fecha
    #df['dt'] = pd.to_datetime(df['dt'])

    # Calcular las estadísticas para cada período de tiempo
    general_stats = calculate_stats(df, 'Country')
    stats_1970_2000 = calculate_stats(df[df['dt'].dt.year.between(1970, 2000)], 'Country')
    stats_2000_2013 = calculate_stats(df[df['dt'].dt.year.between(2000, 2013)], 'Country')

    # Calcular el promedio de la diferencia de temperatura
    average_temp_diff = df.groupby(df['dt'].dt.year)['AverageTemperature'].mean().diff().mean()

    # Organizar todos los datos en un diccionario
    world_data_stats = {
        "general": general_stats,
        "1970_2000": stats_1970_2000,
        "2000_2013": stats_2000_2013,
        "average_temp_diff": average_temp_diff
    }

    return world_data_stats

def start_flask_api():
    try:
        # Create a new process for the Flask API
        subprocess.Popen(["python", "flask_api.py"])
        # Wait for the Flask API to start up
        time.sleep(2)
    except Exception as e:
        st.error(f"Failed to start Flask API: {str(e)}")


# Add this near the beginning of your main code
# Ensure markdown files exist before Streamlit tries to read them
ensure_markdown_files()


# Modify your existing read_markdown_file function to handle errors
def read_markdown_file(markdown_file):
    file_path = Path(markdown_file)
    if not file_path.exists():
        st.warning(f"File not found: {markdown_file}")
        return (
            "# Content Not Available\n\nThe requested markdown file could not be found."
        )
    return file_path.read_text()


# Start Flask API server when the app loads
flask_thread = threading.Thread(target=start_flask_api)
flask_thread.daemon = (
    True  # This ensures the thread will close when the main program exits
)
flask_thread.start()


# Add this function to make requests to the Flask API
def call_flask_api(endpoint, params=None):
    try:
        # Add retry logic for API calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"http://localhost:5000/api/{endpoint}", params=params, timeout=5
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    st.error(f"API Error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retrying
                    else:
                        return None
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    st.warning(f"Connection attempt {attempt+1} failed. Retrying...")
                    time.sleep(2)  # Wait longer before retrying
                else:
                    raise
    except Exception as e:
        st.error(f"API Connection Error: {str(e)}")
        return None


@st.cache_data()
def load_data():

    # World Data
    world_path = os.path.join(os.getcwd(), "Data", "WorldData.csv")
    world_data = pd.read_csv(world_path)
    world_data['dt'] = pd.to_datetime(world_data['dt'])
    world_data['Year'] = world_data['dt'].dt.year

    # USA Data
    usa_filter = world_data['Country']=='United States'
    usa_data = world_data[usa_filter]

    # World Stats
    world_data_stats = get_stats(world_data)
    return world_data, usa_data, world_data_stats


with st.sidebar:
    selected = option_menu(
        menu_title = "",
        options = ["Home","Data", "Model Performance", "Forecasting"],
        icons = ["file-earmark-bar-graph-fill","window-dock", "activity"],
        menu_icon = "image-alt",
        default_index = 0,
        orientation = "vertical",
    )

    # Carga los datos la primera vez que se ejecuta la aplicación
    with st.sidebar:
        loading_message = st.empty()
        loading_message.info('Loading Data...')
        world_data, usa_data, world_data_stats = load_data()
        loading_message.empty()


    if st.button('Refresh', type='secondary'):
        st.cache_data.clear()


col1,col2,col3 = st.columns([1,14,1])
with col2:
    st.title("**🪐NASA's Mission: Pale :blue[Blue Dot]🌎**")

st.image(image1, use_column_width=True)
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.write("Our solution for the NASA mission focuses on the Sustainable Development Goal related to climate action. To achieve this goal, we built a set of predictive models using Prophet, a tool from Meta AI, based on an Earth land surface temperature dataset from 1900 to 2013 published by Berkley Earth. In addition to these predictive models, we conducted a thorough data analysis, which included the creation of plots, geographic maps, and the calculation of descriptive statistics that helped us unravel the complex behavior of Earth's temperature. The proposed models, the data used, the methodology, and the results of our in-depth analysis are detailed in this software, which the authors have named the World Temperature Viewer.")


if selected == "Home":
    st.header('World Temperature Viewer (WTV)')

    st.write("""**World Temperature Viewer (WTV)** is an application developed as part of the NASA Mission: Pale Blue Dot Visualization Challenge. Its primary aim is to enhance visibility and responsibility concerning climate change and global warming. By offering users an interactive interface to engage with global temperature patterns, **WTV** facilitates a deeper understanding of this pressing issue - an issue that we, as a collective, have created and must resolve. Global warming and climate change **DO EXIST**, ignoring these realities will not make them disappear.""") 

    st.write('#### Why is WTV Usefull? :') 

    st.write('###### 📈 Allows to See Posible Potential Temperature Behaivor Around The Globe')
    st.write('###### 📉 Allows to See The Prior Beheivor of Temperature from The Studied Data')
    st.write('###### 📊 Allows a Comparition Between Past and Future Temeratures Around The Globe')
    st.write('###### 🛠️ Allows Access to Insigthfull Content of How the Forecasting Models Were Builded')

    st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    with st.expander('About',True):
        st.title('About:')
        markdown_path = os.path.join(os.getcwd(), "About.md")
        intro_markdown = read_markdown_file(markdown_path)
        st.markdown(intro_markdown, unsafe_allow_html=True)

    with st.expander('Problem Statement'):
        st.title('Problem Statement:')
        intro_markdown = read_markdown_file("WorldTemperatureViewer/Challenge.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)

if selected=='Data':

    st.title('WTV Data')
    st.write('The Data used during this study was Global Land Temperature By City: which has data from 1900 to 2013. However we will display only data from 1970 to 2013 in order to optimize our aplication since the amount of data is very high and Computationaly Spensive')
    st.write('---')

    with st.expander('Earth Temperature Statistics', True):
        st.subheader("Earth Temperature Statistics")
        st.write('📊 Mainly Statistics, Visualization and Comparision of Earth Temperature Data')
        st.header(' ')
        total_countries = len(world_data['Country'].unique())
        c_above = world_data_stats['general']['countries_above_35_degrees_count']
        c_below = world_data_stats['general']['countries_below_0_degrees_count']
        # thc = world_data_stats['general']['']

        col1,col2 = st.columns([3,3])

        with col1:
            st.write('###### Highest Monthly AVG\n ###### Registered Temperature')
            st.metric(f"{world_data_stats['general']['highest_temp']['country']}: {world_data_stats['general']['highest_temp']['city']}",round(world_data_stats['general']['highest_temp']['temperature'],2),str(world_data_stats['general']['highest_temp']['date']))
            st.write('---')
            st.write('Total of Countries with Monthly Average Temperature Above :red[**40**] Degrees Celsius')
            TC_Pie_Plot(total_countries, c_above, 'Above 40C',1)

        with col2:
            st.write('###### Lowest Monthly AVG\n ###### Registered Temperature')
            st.metric(f"{world_data_stats['general']['lowest_temp']['country']}: {world_data_stats['general']['lowest_temp']['city']}",round(world_data_stats['general']['lowest_temp']['temperature'],2), str(world_data_stats['general']['lowest_temp']['date']))
            st.write('---')
            st.write('Total of Countries with Monthly Average Temperature Below :blue[**-10**] Degrees Celsius')
            TC_Pie_Plot(total_countries, c_below, 'Below -10C',0)
    st.write('---')
    st.header(' ')

    with st.expander('Earth Geographic Chart',True):
        st.subheader('Earth 🗺️ Geographic Chart')
        st.write('Between :green[**1970 - 2000**], and :green[**2000 - 2013**], there is a monthly increase in global average temperature of :orange[**0.03**] degrees Celsius. The accompanying graphs show a steady increase in temperature as we approach 2013.')

        GlobeT_Chart(world_data)

    st.header('')
    with st.expander('United States Geographic Chart',True):
        st.subheader("United States 🗽Geographic Chart")
        st.write('Statistics and Visualization of United States Temperature Data')
        USAT_Chart(usa_data)

    with st.expander("API Data Access", False):
        st.subheader("Temperature Data API")
        st.write("Access temperature data through the Flask API")

        api_col1, api_col2 = st.columns([1, 1])

        with api_col1:
            # Get list of countries from API
            countries_data = call_flask_api("countries")
            if countries_data:
                selected_country = st.selectbox(
                    "Select Country",
                    options=countries_data.get("countries", []),
                    key="api_country",
                )
            else:
                st.warning("Could not load countries from API")
                selected_country = None

        with api_col2:
            if selected_country:
                # Get cities for selected country from API
                cities_data = call_flask_api("cities", {"country": selected_country})
                if cities_data:
                    selected_city = st.selectbox(
                        "Select City",
                        options=cities_data.get("cities", []),
                        key="api_city",
                    )
                else:
                    st.warning("Could not load cities from API")
                    selected_city = None

        if selected_country and selected_city:
            if st.button("Get Temperature Statistics via API"):
                stats = call_flask_api(
                    "temperature/stats",
                    {"country": selected_country, "city": selected_city},
                )

                if stats:
                    st.json(stats)

                    # Create a simple visualization of the API data
                    st.subheader(
                        f"Temperature Range for {selected_city}, {selected_country}"
                    )

                    chart_data = pd.DataFrame(
                        {
                            "Metric": ["Average", "Maximum", "Minimum"],
                            "Temperature (°C)": [
                                stats["average_temperature"],
                                stats["max_temperature"],
                                stats["min_temperature"],
                            ],
                        }
                    )

                    chart = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(x="Metric", y="Temperature (°C)", color="Metric")
                        .properties(width=400)
                    )

                    st.altair_chart(chart, use_container_width=True)
if selected=='Model Performance':

    st.header('Models Performance & Comparison Between Real and Predicted Values')
    st.subheader(' ')
    col1,col2,col3 = st.columns([3,3,3])
    st.subheader('General Models Metrics')
    st.write('- ###### *AVG-MAE:* Average Mean Absolute Error Across all Forecsting Models')
    st.write('- ###### *AVG-MSE:* Average Mean Squared Error Across all Forecasting Models')

    col1,col2 = st.columns([5,5])
    col1.metric('AVG-MAE', 1.53, 'High Performance')
    col2.metric('AVG-MSE', 7.31, 'High Performance')

    st.header(" ")
    st.write('---')
    with st.expander('Comparision',True):  
        st.subheader("Select Cities and Dates")
        col1,col2,col3 = st.columns([2,5,2])
        with col2:
            selected_cities = st.multiselect("Cities", cities)

        def plot_real_vs_forecast(models, city, usacitydf):
            usacitydf = usacitydf.rename(columns={'dt': 'ds'})

            # Asegúrate de que las fechas estén en el formato correcto
            start_date = pd.to_datetime('2010-01-01')
            end_date = pd.to_datetime('2013-12-31')

            # Obtén el modelo para la ciudad especificada
            model = models[city]

            # Filtra el dataframe para la ciudad seleccionada y el rango de fechas
            city_filter = (usacitydf['City'] == city) & (usacitydf['ds'] >= start_date) & (usacitydf['ds'] <= end_date)
            df_city = usacitydf[city_filter]

            # Crea un dataframe con todas las fechas desde start_date hasta end_date
            future_dates = pd.date_range(start=start_date, end=end_date)
            future = pd.DataFrame(future_dates, columns=['ds'])

            # Haz la predicción
            forecast = predict_future(model, df_city, 30)

            # Crea el gráfico con Altair
            real = alt.Chart(df_city).mark_line().encode(
                x='ds:T', 
                y='AverageTemperature:Q', 
                color=alt.value('blue'),
                tooltip=['ds:T', 'AverageTemperature:Q']
            ).properties(title='Real Values')

            real = real.interactive()
            pred = alt.Chart(forecast).mark_line().encode(
                x='ds:T', 
                y='yhat:Q', 
                color=alt.value('green'),
                tooltip=['ds:T', 'yhat:Q']
            )    

            pred = pred.interactive()
            confidence_interval = pred.mark_area(opacity=0.3).encode(
                y='yhat_upper:Q',
                y2='yhat_lower:Q'
            )
            print(df_city)

            chart = real + pred
            chart = chart.properties(title=f'{city} Temperature Comparison {start_date.year}-{end_date.year}')
            # chart = chart.interactive()
            return chart

        for city in selected_cities:
            chart = plot_real_vs_forecast(model, city, usa_data)
            st.altair_chart(chart,use_container_width=True)


if selected == "Forecasting":

    st.header("Temperature Forecasting of United States Cities")

    with st.expander("Forecasting of USA Cities", True):
        st.subheader("Select Cities and Dates")
        col1, col2, col3 = st.columns([3, 3, 3])

        with col1:
            selected_cities = st.multiselect("Cities", cities)
        with col2:
            start_year = st.number_input(
                "Start Year", min_value=2024, max_value=2090, value=2024
            )
        with col3:
            end_year = st.number_input(
                "End Year", min_value=start_year, max_value=2091, value=2024
            )

        def plot_forecast_altair_yearly(models, city, start_year, end_year):
            start_date = pd.to_datetime(f"{start_year}-01-01")
            end_date = pd.to_datetime(f"{end_year}-12-31")

            model = models[city]

            future_dates = pd.date_range(start=start_date, end=end_date)
            future = pd.DataFrame(future_dates, columns=["ds"])

            # Predict future temperatures
            forecast = model.predict(future)

            # Create Altair chart
            base = alt.Chart(forecast).encode(x="ds:T")

            forecast_line = base.mark_line(color="green").encode(y="yhat:Q")
            confidence_interval = base.mark_area(opacity=0.3).encode(
                y="yhat_upper:Q", y2="yhat_lower:Q"
            )

            chart = forecast_line + confidence_interval
            chart = chart.properties(title=f"{city} Temperature Forecasting")
            chart = chart.interactive()

            return chart, forecast  # Returning both chart and forecast data
        def check_ollama_connection():
            ollama_host = os.environ.get('OLLAMA_HOST', 'http://ollama:11434')
            max_retries = 5
            retry_delay = 3
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(f"{ollama_host}/api/tags")
                    if response.status_code == 200:
                        st.success("Connected to Ollama successfully!")
                        return True
                except ConnectionError:
                    st.warning(f"Attempt {attempt+1}/{max_retries}: Waiting for Ollama service...")
                    time.sleep(retry_delay)
            
            st.error("Failed to connect to Ollama after multiple attempts. Some features may not work.")
            return False

# Modified Ollama chat function with better error handling
        def get_ollama_response(prompt, model_name="mistral"):
            try:
                ollama_host = os.environ.get('OLLAMA_HOST', 'http://ollama:11434')
                client = ollama.Client(host=ollama_host)
                response = client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response["message"]["content"]
            except Exception as e:
                st.error(f"Error connecting to Ollama: {e}")
                return "Unable to get AI recommendations at this time. Please try again later."

        for city in selected_cities:
            chart, forecast = plot_forecast_altair_yearly(
                model, city, start_year, end_year
            )
            st.altair_chart(chart, use_container_width=True)

            # Get predicted temperature range
            avg_temp = forecast["yhat"].mean()
            max_temp = forecast["yhat"].max()
            min_temp = forecast["yhat"].min()

            st.write(
                f"**Predicted Temperature Range for {city} ({start_year}-{end_year}):**"
            )
            st.write(f"🔹 **Average:** {avg_temp:.2f}°C")
            st.write(f"🔺 **Maximum:** {max_temp:.2f}°C")
            st.write(f"🔻 **Minimum:** {min_temp:.2f}°C")
            st.write("---")

            # **Ollama LLM Analysis**
            if st.button(f"Analyze {city} Forecast & Mitigation Strategies"):
                prompt = f"""
                The temperature forecast for {city} between {start_year} and {end_year} shows an average of {avg_temp:.2f}°C, 
                a maximum of {max_temp:.2f}°C, and a minimum of {min_temp:.2f}°C.
                What are some strategies to reduce urban heat and improve climate resilience in this city?
                """

                # Query Ollama model
                ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
                client = ollama.Client(host=ollama_host)

                # And modify your existing call:
                #response = client.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
                response_text = get_ollama_response(prompt, "mistral")
                st.subheader(f"AI Recommendations for {city}:")
                st.write(response_text)
                st.write("---")
