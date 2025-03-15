import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import json
from datetime import datetime, timedelta
import openai
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="WindBorne Balloon Tracker", layout="wide")

st.title("WindBorne Balloon Tracker")
st.markdown("Visualize and analyze weather balloon data with AI-powered insights")

def fetch_balloon_data(hours_back=24):
    all_data = []
    errors = []
    
    for hour in range(hours_back):
        url = f"https://a.windbornesystems.com/treasure/{hour:02d}.json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                try:
                    balloon_positions = json.loads(response.text)
                    timestamp = datetime.now() - timedelta(hours=hour)
                    for i, position in enumerate(balloon_positions):
                        if isinstance(position, list) and len(position) == 3:
                            lat, lon, alt = position
                            all_data.append({
                                'latitude': lat,
                                'longitude': lon,
                                'altitude': alt,
                                'timestamp': timestamp,
                                'hours_ago': hour
                            })
                except json.JSONDecodeError as e:
                    errors.append(f"Corrupted data for hour {hour}: {str(e)}")
                    try:
                        clean_text = response.text.split(']]]')[0] + ']]]'
                        balloon_positions = json.loads(clean_text)
                        timestamp = datetime.now() - timedelta(hours=hour)
                        st.warning(f"Attempted recovery for hour {hour} - partial data may be available")
                        for i, position in enumerate(balloon_positions):
                            if isinstance(position, list) and len(position) == 3:
                                lat, lon, alt = position
                                all_data.append({
                                    'latitude': lat,
                                    'longitude': lon,
                                    'altitude': alt,
                                    'timestamp': timestamp,
                                    'hours_ago': hour
                                })
                    except:
                        st.warning(f"Could not recover data for hour {hour}")
            else:
                errors.append(f"Failed to fetch data for hour {hour}: Status code {response.status_code}")
        except Exception as e:
            errors.append(f"Error fetching data for hour {hour}: {str(e)}")
    
    if errors:
        with st.expander(f"Encountered {len(errors)} errors while fetching data"):
            for error in errors:
                st.write(error)
    
    return pd.DataFrame(all_data)

# Revised get_weather_data function using the free Current Weather Data API
def get_weather_data(lat, lon, api_key):
    if not api_key:
        st.error("No OpenWeatherMap API key provided.")
        return None
        
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching weather data for ({lat}, {lon}): Status code {response.status_code}. Response: {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception while fetching weather data for ({lat}, {lon}): {str(e)}")
        return None

# Function to generate AI insights
def generate_ai_insights(balloon_df, api_key):
    if not api_key:
        return "Please provide an OpenAI API key to generate insights."
    
    try:
        summary = {
            "total_balloons": len(balloon_df),
            "hours_of_data": balloon_df['hours_ago'].nunique(),
            "avg_altitude": round(balloon_df['altitude'].mean(), 2),
            "min_altitude": round(balloon_df['altitude'].min(), 2),
            "max_altitude": round(balloon_df['altitude'].max(), 2),
            "altitude_distribution": balloon_df.groupby(pd.cut(balloon_df['altitude'], bins=[0, 5, 10, 15, 20, 25]))['altitude'].count().to_dict(),
            "geographic_concentration": balloon_df.groupby([(balloon_df['latitude'] // 10) * 10, (balloon_df['longitude'] // 10) * 10]).size().nlargest(5).to_dict()
        }
        
        client = openai.OpenAI(api_key=api_key)
        prompt = f"""
        Analyze the following weather balloon constellation data and provide operational insights:
        
        Total balloon positions: {summary['total_balloons']}
        Hours of data: {summary['hours_of_data']}
        Average altitude: {summary['avg_altitude']} km
        Altitude range: {summary['min_altitude']} - {summary['max_altitude']} km
        
        Altitude distribution:
        {summary['altitude_distribution']}
        
        Top 5 geographic regions with highest balloon concentration:
        {summary['geographic_concentration']}
        
        Based on this data:
        1. Provide 3-4 key operational insights for balloon deployment strategy
        2. Identify any patterns or anomalies in the data
        3. Suggest how this data could be used to improve weather forecasting
        4. Recommend optimal altitude ranges for future deployments
        
        Keep your analysis concise and focused on actionable insights.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a weather balloon operations analyst providing insights from constellation data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating insights: {str(e)}"

st.sidebar.header("Settings")
hours_to_display = st.sidebar.slider("Hours to display", 1, 24, 12)
skip_corrupted = st.sidebar.checkbox("Skip corrupted data sources", value=True)
refresh = st.sidebar.button("Refresh Data")

with st.sidebar.expander("API Keys (Optional)"):
    openweather_key = st.text_input("OpenWeatherMap API Key", type="password")
    openai_key = st.text_input("OpenAI API Key", type="password")
    st.info("Your API keys are not stored and will only be used for this session.")

st.sidebar.markdown("---")
st.sidebar.subheader("Data Status")
status_container = st.sidebar.empty()

if 'balloon_df' not in st.session_state or refresh:
    with st.spinner("Fetching balloon data..."):
        balloon_df = fetch_balloon_data(hours_to_display)
        if not balloon_df.empty:
            st.session_state.balloon_df = balloon_df
            status_container.success(f"Successfully loaded {len(balloon_df)} balloon positions from {balloon_df['hours_ago'].nunique()} hours")
        else:
            st.error("No balloon data available.")
            status_container.error("Failed to load data")

balloon_df = st.session_state.get('balloon_df', pd.DataFrame())

if not balloon_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Balloon Positions", len(balloon_df))
    col2.metric("Hours of Data", balloon_df['hours_ago'].nunique())
    col3.metric("Average Altitude", f"{balloon_df['altitude'].mean():.2f} km")
    
    st.subheader("Balloon Positions")

clean_balloon_df = balloon_df.dropna(subset=['latitude', 'longitude'])

if len(clean_balloon_df) > 0:
    m = folium.Map(location=[clean_balloon_df['latitude'].mean(), clean_balloon_df['longitude'].mean()], 
                  zoom_start=3)
    
    for _, row in clean_balloon_df.iterrows():
        normalized_alt = min(1.0, row['altitude'] / 25.0)  
        color = f'#{int(255*normalized_alt):02x}{0:02x}{int(255*(1-normalized_alt)):02x}'
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"Altitude: {row['altitude']:.2f}km, Hours ago: {row['hours_ago']}"
        ).add_to(m)
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 90px; 
    border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
    padding: 10px; border-radius: 5px;">
    <p><span style="background-color: #0000ff; width: 15px; height: 15px; display: inline-block;"></span> Low altitude</p>
    <p><span style="background-color: #ff0000; width: 15px; height: 15px; display: inline-block;"></span> High altitude</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    st_data = folium_static(m, width=1200, height=500)
    
    if len(balloon_df) > len(clean_balloon_df):
        st.warning(f"Note: {len(balloon_df) - len(clean_balloon_df)} balloon positions were filtered out due to missing coordinates.")

    tab1, tab2, tab3 = st.tabs(["Altitude Analysis", "Weather Comparison", "AI Assistant"])
    
    with tab1:
        st.subheader("Altitude Distribution")
        alt_fig = px.histogram(
            balloon_df, 
            x="altitude", 
            nbins=20,
            title="Balloon Altitude Distribution",
            labels={"altitude": "Altitude (km)", "count": "Count"}
        )
        alt_fig.update_layout(xaxis=dict(tickmode='linear', dtick=5))
        st.plotly_chart(alt_fig, use_container_width=True)
        
        st.subheader("Altitude vs. Time")
        time_fig = px.scatter(
            balloon_df,
            x="hours_ago",
            y="altitude",
            color="altitude",
            title="Balloon Altitude vs. Time",
            labels={"hours_ago": "Hours Ago", "altitude": "Altitude (km)"}
        )
        time_fig.update_layout(xaxis=dict(tickmode='linear', dtick=1), yaxis=dict(tickmode='linear', dtick=5))
        st.plotly_chart(time_fig, use_container_width=True)
    
    with tab2:
     if openweather_key:
        st.subheader("Surface Weather Comparison")
        
        st.write("Select a method to view weather data:")
        selection_method = st.radio(
            "Selection method:",
            ["View Recent Locations", "Click on Map", "Cycle Through Points"],
            horizontal=True
        )
        
        if selection_method == "View Recent Locations":
            sample_size = min(5, len(balloon_df))
            if sample_size > 0:
                recent_df = balloon_df[balloon_df['hours_ago'] == balloon_df['hours_ago'].min()]
                sample_locations = recent_df.sample(sample_size)
                
                weather_data = []
                for _, location in sample_locations.iterrows():
                    lat, lon = location['latitude'], location['longitude']
                    weather = get_weather_data(lat, lon, openweather_key)
                    if weather:
                        weather_data.append({
                            'latitude': lat,
                            'longitude': lon,
                            'altitude': location['altitude'],
                            'temperature': weather.get('main', {}).get('temp'),
                            'wind_speed': weather.get('wind', {}).get('speed'),
                            'conditions': weather.get('weather', [{}])[0].get('description', 'Unknown')
                        })
                
        elif selection_method == "Click on Map":
            st.write("Click on a point in the map to see weather data:")
            
            clean_balloon_df = balloon_df.dropna(subset=['latitude', 'longitude'])
            
            selection_fig = px.scatter_geo(
                clean_balloon_df,
                lat="latitude",
                lon="longitude",
                color="altitude",
                projection="natural earth",
                title="Click to select a balloon"
            )
            selection_fig.update_traces(marker=dict(size=10))
            selection_fig.update_layout(height=400)
            
            selected_point = st.plotly_chart(selection_fig, use_container_width=True, key="selection_map")
            
            if selected_point is not None and hasattr(selected_point, 'get'):
                clicked_point = selected_point.get('points', [{}])[0]
                if clicked_point:
                    lat = clicked_point.get('lat')
                    lon = clicked_point.get('lon')
                    if lat is not None and lon is not None:
                        st.write(f"Selected point at: {lat:.4f}, {lon:.4f}")
                        weather = get_weather_data(lat, lon, openweather_key)
                        if weather:
                            weather_data = [{
                                'latitude': lat,
                                'longitude': lon,
                                'altitude': clicked_point.get('customdata', [0])[0],
                                'temperature': weather.get('main', {}).get('temp'),
                                'wind_speed': weather.get('wind', {}).get('speed'),
                                'conditions': weather.get('weather', [{}])[0].get('description', 'Unknown')
                            }]
                        else:
                            st.warning("Could not fetch weather data for selected point.")
                            weather_data = []
                    else:
                        st.info("Click on a balloon point to see weather data.")
                        weather_data = []
                else:
                    st.info("Click on a balloon point to see weather data.")
                    weather_data = []
            else:
                st.info("Click on a balloon point to see weather data.")
                weather_data = []
                
        elif selection_method == "Cycle Through Points":
            clean_balloon_df = balloon_df.dropna(subset=['latitude', 'longitude'])
            
            if len(clean_balloon_df) > 0:
                cycle_index = st.slider("Cycle through points", 0, len(clean_balloon_df)-1, 0)
                selected_point = clean_balloon_df.iloc[cycle_index]
                
                st.write(f"Selected point {cycle_index+1}/{len(clean_balloon_df)}")
                st.write(f"Latitude: {selected_point['latitude']:.4f}, Longitude: {selected_point['longitude']:.4f}")
                st.write(f"Altitude: {selected_point['altitude']:.2f} km")
                
                weather = get_weather_data(selected_point['latitude'], selected_point['longitude'], openweather_key)
                if weather:
                    weather_data = [{
                        'latitude': selected_point['latitude'],
                        'longitude': selected_point['longitude'],
                        'altitude': selected_point['altitude'],
                        'temperature': weather.get('main', {}).get('temp'),
                        'wind_speed': weather.get('wind', {}).get('speed'),
                        'conditions': weather.get('weather', [{}])[0].get('description', 'Unknown')
                    }]
                else:
                    st.warning("Could not fetch weather data for selected point.")
                    weather_data = []
            else:
                st.warning("No valid balloon positions available for cycling.")
                weather_data = []
        
        if weather_data:
            weather_df = pd.DataFrame(weather_data)
            st.dataframe(weather_df)
            
            weather_map = px.scatter_mapbox(
                weather_df,
                lat="latitude",
                lon="longitude",
                color="temperature",
                size="wind_speed",
                zoom=3,
                height=300,
                title="Surface Weather at Selected Balloon Locations"
            )
            weather_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(weather_map, use_container_width=True)
            
            st.subheader("Weather Insights")
            avg_temp = weather_df['temperature'].mean()
            avg_wind = weather_df['wind_speed'].mean()
            st.write(f"Average surface temperature: {avg_temp:.1f}°C")
            st.write(f"Average surface wind speed: {avg_wind:.1f} m/s")
            st.write("Conditions:")
            for condition in weather_df['conditions'].unique():
                st.write(f"- {condition}")
        else:
            st.info("No weather data available for the selected points.")
     else:
        st.info("Enter an OpenWeatherMap API key in the sidebar to see weather comparison.")
    
    with tab3:
        st.subheader("AI-Powered Balloon Analysis")
        if openai_key:
            st.write("#### Operational Insights")
            if 'ai_insights' not in st.session_state or refresh:
                with st.spinner("Generating AI insights..."):
                    st.session_state.ai_insights = generate_ai_insights(balloon_df, openai_key)
            st.markdown(st.session_state.ai_insights)
            
            st.write("#### Ask a Question About the Balloon Data")
            st.write("Examples: What altitudes are most common? Where are most balloons located? What patterns do you see in the data?")
            user_question = st.text_input("Your question:")
            if user_question:
                with st.spinner("Analyzing data..."):
                    try:
                        client = openai.OpenAI(api_key=openai_key)
                        data_context = {
                            "total_balloons": len(balloon_df),
                            "hours_of_data": balloon_df['hours_ago'].nunique(),
                            "avg_altitude": round(balloon_df['altitude'].mean(), 2),
                            "altitude_range": f"{round(balloon_df['altitude'].min(), 2)} - {round(balloon_df['altitude'].max(), 2)} km",
                            "altitude_distribution": balloon_df.groupby(pd.cut(balloon_df['altitude'], bins=[0, 5, 10, 15, 20, 25]))['altitude'].count().to_dict(),
                            "geographic_coverage": f"Latitude: {round(balloon_df['latitude'].min(), 2)} to {round(balloon_df['latitude'].max(), 2)}, Longitude: {round(balloon_df['longitude'].min(), 2)} to {round(balloon_df['longitude'].max(), 2)}"
                        }
                        prompt = f"""
                        Here is data about a weather balloon constellation:
                        
                        Total balloon positions: {data_context['total_balloons']}
                        Hours of data: {data_context['hours_of_data']}
                        Average altitude: {data_context['avg_altitude']} km
                        Altitude range: {data_context['altitude_range']}
                        Geographic coverage: {data_context['geographic_coverage']}
                        
                        Based on this data, please answer the following question:
                        {user_question}
                        
                        If the question cannot be answered with the available data, please explain what additional data would be needed.
                        """
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a weather balloon operations expert answering questions about constellation data."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=400
                        )
                        answer = response.choices[0].message.content
                    except Exception as e:
                        answer = f"Error processing your question: {str(e)}"
                st.markdown(answer)
        else:
            st.info("Enter an OpenAI API key in the sidebar to use the AI Assistant features.")

    st.header("Operational Insights")
    altitude_bands = pd.cut(balloon_df['altitude'], bins=[0, 5, 10, 15, 20, 25])
    altitude_counts = balloon_df.groupby(altitude_bands).size().reset_index(name='count')
    altitude_counts.columns = ['Altitude Range', 'Count']
    st.subheader("Balloon Distribution by Altitude")
    st.write("Most balloons are concentrated in these altitude ranges:")
    st.dataframe(altitude_counts)
    
    st.subheader("Geographic Concentration")
    balloon_df['lat_region'] = (balloon_df['latitude'] // 10) * 10
    balloon_df['lon_region'] = (balloon_df['longitude'] // 10) * 10
    region_counts = balloon_df.groupby(['lat_region', 'lon_region']).size().reset_index(name='count')
    region_counts = region_counts.sort_values('count', ascending=False).head(5)
    st.write("Top 5 regions with highest balloon concentration:")
    for _, row in region_counts.iterrows():
        st.write(f"- Region {row['lat_region']}° to {row['lat_region']+10}° latitude, {row['lon_region']}° to {row['lon_region']+10}° longitude: {row['count']} balloons")
else:
    st.error("No data available. Please refresh the data.")

st.sidebar.markdown("---")
st.sidebar.subheader("Project Notes")
st.sidebar.markdown("""
I created this dashboard to analyze WindBorne's balloon constellation and provide operational insights by combining high-altitude balloon data with surface weather conditions and AI-powered analysis. This tool helps visualize the distribution of balloons across different altitudes and geographic regions, while the LLM integration provides deeper insights and allows users to ask specific questions about the data.
""")
