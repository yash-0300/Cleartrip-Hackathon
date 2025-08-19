import streamlit as st
import json
from datetime import datetime

from agents import (
    fetch_flights, 
    extract_cheapest_flights, 
    format_datetime, 
    researcher, 
    planner, 
    hotel_restaurant_finder,
    fetch_booking_token
)

# Set up Streamlit UI with a travel-friendly theme
st.set_page_config(page_title="🌍 Cleartrip Travel Planner", layout="wide")
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ff5733;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555;
        }
        .stSlider > div {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and subtitle
st.markdown('<h1 class="title">✈️ Cleartrip-Powered Travel Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plan your dream trip with Cleartrip! Get personalized recommendations for flights, hotels, and activities.</p>', unsafe_allow_html=True)

# User Inputs Section
st.markdown("### 🌍 Where are you headed?")
source = st.text_input("🛫 Departure City (IATA Code):", "BOM")  # Example: BOM for Mumbai
destination = st.text_input("🛬 Destination (IATA Code):", "DEL")  # Example: DEL for Delhi

st.markdown("### 📅 Plan Your Adventure")
num_days = st.slider("🕒 Trip Duration (days):", 1, 14, 3)
travel_theme = st.selectbox(
    "🎭 Select Your Travel Theme:",
    ["👨‍👩‍👧‍👦 Family Vacation", "💑 Couple Getaway", "🏔️ Adventure Trip", "🧳 Solo Exploration"]
)

# Divider for aesthetics
st.markdown("---")

st.markdown(
    f"""
    <div style="
        text-align: center; 
        padding: 15px; 
        background-color: #ffecd1;
        border-radius: 10px; 
        margin-top: 20px;
    ">
        <h3 style="color: black;">🌟 Your {travel_theme} to {destination} is about to begin! 🌟</h3>
        <p style="color: black;">Let's find the best flights, stays, and experiences for your unforgettable journey.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

activity_preferences = st.text_area(
    "🌍 What activities do you enjoy? (e.g., relaxing on the beach, exploring historical sites, nightlife, adventure)",
    "Relaxing on the beach, exploring historical sites"
)

departure_date = st.date_input("Departure Date")
return_date = st.date_input("Return Date")

# Sidebar Setup
st.sidebar.title("🌎 Travel Assistant")
st.sidebar.subheader("Personalize Your Trip")

# Travel Preferences
budget = st.sidebar.radio("💰 Budget Preference:", ["Economy", "Standard", "Luxury"])
flight_class = st.sidebar.radio("✈️ Flight Class:", ["Economy", "Business", "First Class"])
hotel_rating = st.sidebar.selectbox("🏨 Preferred Hotel Rating:", ["Any", "3⭐", "4⭐", "5⭐"])

# Packing Checklist
st.sidebar.subheader("🎒 Packing Checklist")
packing_list = {
    "👕 Clothes": True,
    "🩴 Comfortable Footwear": True,
    "🕶️ Sunglasses & Sunscreen": False,
    "📖 Travel Guidebook": False,
    "💊 Medications & First-Aid": True
}
for item, checked in packing_list.items():
    st.sidebar.checkbox(item, value=checked)

# Travel Essentials
st.sidebar.subheader("🛂 Travel Essentials")
visa_required = st.sidebar.checkbox("🛃 Check Visa Requirements")
travel_insurance = st.sidebar.checkbox("🛡️ Get Travel Insurance")
currency_converter = st.sidebar.checkbox("💱 Currency Exchange Rates")

# Generate Travel Plan
if st.button("🚀 Generate Travel Plan"):
    with st.spinner("✈️ Fetching best flight options..."):
        flight_data = fetch_flights(source, destination, departure_date, return_date)
        cheapest_flights = extract_cheapest_flights(flight_data)

    # AI Processing
    with st.spinner("🔍 Researching best attractions & activities..."):
        research_prompt = (
            f"Research the best attractions and activities in {destination} for a {num_days}-day {travel_theme.lower()} trip. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. "
            f"Hotel Rating: {hotel_rating}. Visa Requirement: {visa_required}. Travel Insurance: {travel_insurance}."
        )
        research_results = researcher.run(research_prompt, stream=False)

    with st.spinner("🏨 Searching for hotels & restaurants..."):
        hotel_restaurant_prompt = (
            f"Find the best hotels and restaurants near popular attractions in {destination} for a {travel_theme.lower()} trip. "
            f"Budget: {budget}. Hotel Rating: {hotel_rating}. Preferred activities: {activity_preferences}."
        )
        hotel_restaurant_results = hotel_restaurant_finder.run(hotel_restaurant_prompt, stream=False)

    with st.spinner("🗺️ Creating your personalized itinerary..."):
        planning_prompt = (
            f"Based on the following data, create a {num_days}-day itinerary for a {travel_theme.lower()} trip to {destination}. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. Hotel Rating: {hotel_rating}. "
            f"Visa Requirement: {visa_required}. Travel Insurance: {travel_insurance}. Research: {research_results.content}. "
            f"Flights: {json.dumps(cheapest_flights)}. Hotels & Restaurants: {hotel_restaurant_results.content}."
        )
        itinerary = planner.run(planning_prompt, stream=False)

    # Display Results
    st.subheader("✈️ Cheapest Flight Options")
    if cheapest_flights:
        cols = st.columns(len(cheapest_flights))
        for idx, flight in enumerate(cheapest_flights):
            with cols[idx]:
                airline_logo = flight.get("airline_logo", "")
                airline_name = flight.get("airline", "Unknown Airline")
                price = flight.get("price", "Not Available")
                total_duration = flight.get("total_duration", "N/A")
                
                flights_info = flight.get("flights", [{}])
                departure = flights_info[0].get("departure_airport", {})
                arrival = flights_info[-1].get("arrival_airport", {})
                airline_name = flights_info[0].get("airline", "Unknown Airline") 
                
                departure_time = format_datetime(departure.get("time", "N/A"))
                arrival_time = format_datetime(arrival.get("time", "N/A"))
                
                booking_token = fetch_booking_token(flight, flight_data)
                booking_link = f"https://www.google.com/travel/flights?tfs={booking_token}" if booking_token else "#"
                
                # Flight card layout
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd; 
                        border-radius: 10px; 
                        padding: 15px; 
                        text-align: center;
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                        background-color: #f9f9f9;
                        margin-bottom: 20px;
                    ">
                        <img src="{airline_logo}" width="100" alt="Flight Logo" />
                        <h3 style="margin: 10px 0;">{airline_name}</h3>
                        <p><strong>Departure:</strong> {departure_time}</p>
                        <p><strong>Arrival:</strong> {arrival_time}</p>
                        <p><strong>Duration:</strong> {total_duration} min</p>
                        <h2 style="color: #008000;">💰 {price}</h2>
                        <a href="{booking_link}" target="_blank" style="
                            display: inline-block;
                            padding: 10px 20px;
                            font-size: 16px;
                            font-weight: bold;
                            color: #fff;
                            background-color: #007bff;
                            text-decoration: none;
                            border-radius: 5px;
                            margin-top: 10px;
                        ">🔗 Book Now</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("⚠️ No flight data available.")

    st.subheader("🏨 Hotels & Restaurants")
    st.write(hotel_restaurant_results.content)

    st.subheader("🗺️ Your Personalized Itinerary")
    st.write(itinerary.content)

    st.success("✅ Travel plan generated successfully!")