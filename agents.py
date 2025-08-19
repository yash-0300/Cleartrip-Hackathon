import json
import os
import serpapi
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.google import Gemini
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def format_datetime(iso_string):
    """Formats an ISO datetime string to a more readable format."""
    try:
        dt = datetime.strptime(iso_string, "%Y-%m-%d %H:%M")
        return dt.strftime("%b-%d, %Y | %I:%M %p")
    except (ValueError, TypeError):
        return "N/A"

def fetch_flights(source, destination, departure_date, return_date):
    """Fetches flight data from the Google Flights API via SerpApi."""
    params = {
        "engine": "google_flights",
        "departure_id": source,
        "arrival_id": destination,
        "outbound_date": str(departure_date),
        "return_date": str(return_date),
        "currency": "INR",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }
    try:
        search = serpapi.search(params)
        results = search.as_dict()
        return results
    except Exception as e:
        print(f"Error fetching flights: {e}")
        return {}

def extract_cheapest_flights(flight_data):
    """Extracts the top 3 cheapest flights from the flight data."""
    best_flights = flight_data.get("best_flights", [])
    if not best_flights:
        return []
    sorted_flights = sorted(best_flights, key=lambda x: x.get("price", float("inf")))[:3]
    return sorted_flights

def fetch_booking_token(flight_details, params):
    """Fetches a booking token for a specific flight."""
    try:
        departure_token = flight_details.get("departure_token", "")
        if departure_token:
            params_with_token = {
                **params,
                "departure_token": departure_token
            }
            search_with_token = serpapi.search(params_with_token)
            results_with_booking = search_with_token.as_dict()
            booking_token = results_with_booking['best_flights'][0]['booking_token']
            return booking_token
    except Exception as e:
        print(f"Error fetching booking token: {e}")
        return None

# Initialize AI Agents
researcher = Agent(
    name="Researcher",
    instructions=[
        "Identify the travel destination specified by the user.",
        "Gather detailed information on the destination, including climate, culture, and safety tips.",
        "Find popular attractions, landmarks, and must-visit places.",
        "Search for activities that match the user’s interests and travel style.",
        "Prioritize information from reliable sources and official travel guides.",
        "Provide well-structured summaries with key insights and recommendations."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

planner = Agent(
    name="Planner",
    instructions=[
        "Gather details about the user's travel preferences and budget.",
        "Create a detailed itinerary with scheduled activities and estimated costs.",
        "Ensure the itinerary includes transportation options and travel time estimates.",
        "Optimize the schedule for convenience and enjoyment.",
        "Present the itinerary in a structured format."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    add_datetime_to_instructions=True,
)

hotel_restaurant_finder = Agent(
    name="Hotel & Restaurant Finder",
    instructions=[
        "Identify key locations in the user's travel itinerary.",
        "Search for highly rated hotels near those locations.",
        "Search for top-rated restaurants based on cuisine preferences and proximity.",
        "Prioritize results based on user preferences, ratings, and availability.",
        "Provide direct booking links or reservation options where possible."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)