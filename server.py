from flask import Flask, render_template, jsonify, request
from markupsafe import Markup
import os
import base64
import json
import datetime
import time
import logging
from dotenv import load_dotenv
import requests
import markdown
from pymdownx import superfences, highlight
from flask_talisman import Talisman

# Load environment variables
load_dotenv()

# Configure logging - use only stream handler for Vercel compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure Talisman for security headers
csp = {
    'default-src': "'self'",
    'script-src': "'self' 'unsafe-inline'",
    'connect-src': "'self'",
    'img-src': "'self' data: https://openweathermap.org",
    'style-src': "'self' 'unsafe-inline'"
}
Talisman(app, content_security_policy=csp, force_https=False)  # Set force_https=True in production

# Get API key from environment variable
API_KEY = os.environ.get('TOGETHER_API_KEY')
# Check if we're in demo mode
DEMO_MODE = os.environ.get('DEMO_MODE', 'False').lower() in ('true', '1', 'yes')
if not API_KEY and not DEMO_MODE:
    raise ValueError("TOGETHER_API_KEY environment variable is not set. Set DEMO_MODE=True to run without an API key.")

# Define valid models with their specific settings
# Together API constraint: (prompt/input tokens + max_tokens) <= 8193
# Do NOT raise max_tokens above 8192 unless Together raises their total context window.
VALID_MODELS = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": {
        "name": "Llama 3.3 70B",
        "temperature": 0.7,
        "max_tokens": 2048,  # Safe for most prompts; user+output never exceeds 8193
        "max_tokens_quick": 256,  # Also keep "quick" within safe API limits
        "top_p": 0.9,
        "supports_functions": True
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free": {
        "name": "DeepSeek R1",
        "temperature": 0.6,  # Recommended temperature for DeepSeek R1
        "max_tokens": 2048,  # Safe for most prompts; user+output never exceeds 8193
        "max_tokens_quick": 256,  # Also keep "quick" within safe API limits
        "top_p": 0.95,  # Recommended top_p for DeepSeek R1
        "supports_functions": False
    }
}

# Define available functions for function calling
AVAILABLE_FUNCTIONS = {
    "get_real_weather": {
        "description": "Get real-time weather data for a location using OpenWeatherMap API with current conditions, temperature, humidity, wind, and more",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. London, Tokyo, New York"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature to use. Infer this from the user's location."
                }
            },
            "required": ["location"]
        }
    },
    "get_real_time": {
        "description": "Get accurate real-time data for a location using TimeZoneDB API with precise timezone information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the current time for, e.g. 'New York' or 'Tokyo'"
                },
                "format": {
                    "type": "string",
                    "enum": ["12h", "24h"],
                    "description": "The time format to use (12-hour or 24-hour)"
                }
            },
            "required": ["location"]
        }
    },
    "get_weather": {
        "description": "Get simulated weather data (only use if real-time data is unavailable)",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature to use. Infer this from the user's location."
                }
            },
            "required": ["location"]
        }
    },
    "get_current_time": {
        "description": "Get simulated time data (only use if real-time data is unavailable)",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the current time for, e.g. 'New York' or 'Tokyo'"
                },
                "format": {
                    "type": "string",
                    "enum": ["12h", "24h"],
                    "description": "The time format to use (12-hour or 24-hour)"
                }
            },
            "required": ["location"]
        }
    },
    "search_products": {
        "description": "Search for products in an e-commerce catalog",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for products"
                },
                "category": {
                    "type": "string",
                    "description": "The category to filter by (optional)"
                },
                "max_price": {
                    "type": "number",
                    "description": "The maximum price to filter by (optional)"
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["price_asc", "price_desc", "popularity", "rating"],
                    "description": "How to sort the results (optional)"
                }
            },
            "required": ["query"]
        }
    },
    "search_wikipedia": {
        "description": "Search Wikipedia for information on a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or topic to look up on Wikipedia"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 1)",
                    "default": 1
                }
            },
            "required": ["query"]
        }
    }
}

@app.route('/')
def index():
    return render_template('snapenhance.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return the list of available models with their settings"""
    return jsonify({
        "models": [
            {
                "id": model_id,
                "name": model_settings["name"],
                "temperature": model_settings["temperature"],
                "max_tokens": model_settings["max_tokens"],
                "top_p": model_settings["top_p"]
            }
            for model_id, model_settings in VALID_MODELS.items()
        ]
    })

# Function implementations
def execute_function(function_name, arguments):
    """Execute the requested function with the provided arguments"""
    # Clean up function name - sometimes the model returns the description instead of the name
    function_name = function_name.lower()

    # Check if this is a greeting or simple query that shouldn't use functions
    greeting_terms = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon",
                     "good evening", "how are you", "what's up", "nice to meet you"]

    # Get the query from arguments if it exists (for search functions)
    query = ""
    if "query" in arguments:
        query = str(arguments.get("query", "")).lower()
    elif "location" in arguments:
        query = str(arguments.get("location", "")).lower()

    # Check if the query is just a greeting
    if any(term == query for term in greeting_terms):
        logger.warning(f"Prevented function call for simple greeting: {query}")
        return {
            "error": "Function calls are not needed for simple greetings",
            "message": "This is a simple greeting that doesn't require API data"
        }

    # Map function descriptions to actual function names
    # The first function in each list is the preferred implementation
    function_map = {
        # Always prefer real-time weather data
        "weather": {
            "aliases": ["weather", "get weather", "current weather", "get the current weather", "get_weather", "get_real_weather", "real weather", "real-time weather"],
            "function": "get_real_weather",
            "fallback": "get_weather"
        },
        # Always prefer real-time time data
        "time": {
            "aliases": ["time", "current time", "get time", "get the current time", "get_current_time", "get_real_time", "accurate time", "real time"],
            "function": "get_real_time",
            "fallback": "get_current_time"
        },
        "products": {
            "aliases": ["products", "search products", "search for products", "get products", "find products", "search_products"],
            "function": "search_products",
            "fallback": None
        },
        "wikipedia": {
            "aliases": ["wikipedia search", "search wikipedia", "wiki search", "search wiki", "search_wikipedia", "lookup wikipedia", "find on wikipedia", "wikipedia article"],
            "function": "search_wikipedia",
            "fallback": None
        }
    }

    # Find the matching function category
    matched_category = None
    matched_function = None

    for category, config in function_map.items():
        if any(alias in function_name for alias in config["aliases"]):
            matched_category = category
            matched_function = config["function"]
            break

    logger.info(f"Mapped function '{function_name}' to category '{matched_category}' with function '{matched_function}'")

    # Execute the appropriate function with real-time data priority
    if matched_category == "weather":
        # Always try real-time weather first
        result = get_real_weather(arguments.get("location"), arguments.get("unit", "celsius"))
        # Only fall back to simulated data if there was an API error and we have no API key
        if "error" in result and "API key not provided" in result.get("error", ""):
            logger.info("Falling back to simulated weather data due to missing API key")
            return get_weather(arguments.get("location"), arguments.get("unit", "celsius"))
        return result
    elif matched_category == "time":
        # Always try real-time time first
        result = get_real_time(arguments.get("location"), arguments.get("format", "24h"))
        # Only fall back if there was an API error and we have no API keys
        if "error" in result or "note" in result and "API key not provided" in result.get("note", ""):
            logger.info("Falling back to simulated time data due to missing API keys")
            return get_current_time(arguments.get("location"), arguments.get("format", "24h"))
        return result
    elif matched_category == "products":
        return search_products(
            arguments.get("query"),
            arguments.get("category"),
            arguments.get("max_price"),
            arguments.get("sort_by", "popularity")
        )
    elif matched_category == "wikipedia":
        return search_wikipedia(
            arguments.get("query"),
            arguments.get("limit", 1)
        )
    else:
        return {"error": f"Function {function_name} not implemented"}

def get_weather(location, unit="celsius"):
    """Simulate getting weather data for a location"""
    # In a real implementation, this would call a weather API
    weather_data = {
        "New York": {"temp": 22, "condition": "Partly Cloudy", "humidity": 65},
        "London": {"temp": 18, "condition": "Rainy", "humidity": 80},
        "Tokyo": {"temp": 26, "condition": "Sunny", "humidity": 70},
        "Sydney": {"temp": 24, "condition": "Clear", "humidity": 60},
        "Paris": {"temp": 20, "condition": "Cloudy", "humidity": 75},
    }

    # Default weather if location not found
    default_weather = {"temp": 23, "condition": "Sunny", "humidity": 68}

    # Get weather for the location or use default
    weather = weather_data.get(location, default_weather)

    # Convert temperature if needed
    temp = weather["temp"]
    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32

    return {
        "location": location,
        "temperature": f"{round(temp)}°{'F' if unit == 'fahrenheit' else 'C'}",
        "condition": weather["condition"],
        "humidity": f"{weather['humidity']}%",
        "unit": unit,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def get_real_weather(location, unit="celsius"):
    """Get real weather data from OpenWeatherMap API"""
    api_key = os.environ.get('OPENWEATHERMAP_API_KEY')

    # If no API key is provided, return an error
    if not api_key or api_key == "your_openweathermap_api_key":
        return {
            "error": "OpenWeatherMap API key not provided",
            "location": location,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # Set units parameter for API request
    units = "metric" if unit == "celsius" else "imperial"

    try:
        # Make API request
        print(f"Getting weather data for '{location}' using OpenWeatherMap API")
        response = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={location}&units={units}&appid={api_key}",
            timeout=5
        )

        # Check if request was successful
        if response.status_code == 200:
            data = response.json()

            # Extract relevant data
            temp_symbol = "°C" if unit == "celsius" else "°F"
            speed_unit = "mph" if unit == "fahrenheit" else "m/s"

            # Format sunrise and sunset times
            sunrise_time = datetime.datetime.fromtimestamp(data['sys']['sunrise'])
            sunset_time = datetime.datetime.fromtimestamp(data['sys']['sunset'])

            # Get weather icon URL
            icon_code = data['weather'][0]['icon']
            icon_url = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"

            return {
                "location": f"{data['name']}, {data['sys']['country']}",
                "coordinates": f"Lat: {data['coord']['lat']}, Lon: {data['coord']['lon']}",
                "temperature": f"{round(data['main']['temp'])}{temp_symbol}",
                "feels_like": f"{round(data['main']['feels_like'])}{temp_symbol}",
                "condition": data['weather'][0]['main'],
                "description": data['weather'][0]['description'].capitalize(),
                "icon_url": icon_url,
                "humidity": f"{data['main']['humidity']}%",
                "wind_speed": f"{data['wind']['speed']} {speed_unit}",
                "wind_direction": get_wind_direction(data['wind'].get('deg', 0)),
                "pressure": f"{data['main']['pressure']} hPa",
                "visibility": f"{data.get('visibility', 0) / 1000} km",
                "sunrise": sunrise_time.strftime("%H:%M"),
                "sunset": sunset_time.strftime("%H:%M"),
                "timezone": f"UTC{'+' if data['timezone'] >= 0 else ''}{data['timezone'] / 3600}",
                "unit": unit,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "OpenWeatherMap API (real-time data)"
            }
        else:
            # If API request failed, return error
            error_data = response.json() if response.text else {"message": "Unknown error"}
            error_message = error_data.get('message', f"Error code: {response.status_code}")

            return {
                "error": f"Could not retrieve weather data: {error_message}",
                "location": location,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    except Exception as e:
        # If any exception occurs, return the error
        return {
            "error": f"Error accessing weather API: {str(e)}",
            "location": location,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def get_wind_direction(degrees):
    """Convert wind direction in degrees to cardinal direction"""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / (360 / len(directions))) % len(directions)
    return directions[index]

def get_current_time(location, format="24h"):
    """Simulate getting current time for a location"""
    # In a real implementation, this would use a timezone API
    time_offsets = {
        # North America
        "New York": -4,  # EDT
        "Los Angeles": -7, # PDT
        "Chicago": -5,   # CDT
        "Toronto": -4,   # EDT
        "Mexico City": -6, # CST
        "USA": -5,       # EST/CDT (average)
        "Canada": -5,    # EST/CDT (average)
        "Mexico": -6,    # CST

        # Europe
        "London": 1,     # BST
        "Paris": 2,      # CEST
        "Berlin": 2,     # CEST
        "Rome": 2,       # CEST
        "Madrid": 2,     # CEST
        "Moscow": 3,     # MSK
        "UK": 1,         # BST
        "France": 2,     # CEST
        "Germany": 2,    # CEST
        "Italy": 2,      # CEST
        "Spain": 2,      # CEST
        "Russia": 3,     # MSK (Moscow)

        # Asia
        "Tokyo": 9,      # JST
        "Beijing": 8,    # CST
        "Shanghai": 8,   # CST
        "Mumbai": 5.5,   # IST
        "Delhi": 5.5,    # IST
        "Bangalore": 5.5, # IST
        "Kolkata": 5.5,  # IST
        "Chennai": 5.5,  # IST
        "Hyderabad": 5.5, # IST
        "Singapore": 8,  # SGT
        "Hong Kong": 8,  # HKT
        "Seoul": 9,      # KST
        "Dubai": 4,      # GST
        "Japan": 9,      # JST
        "China": 8,      # CST
        "India": 5.5,    # IST
        "South Korea": 9, # KST
        "UAE": 4,        # GST

        # Oceania
        "Sydney": 10,    # AEST
        "Melbourne": 10, # AEST
        "Brisbane": 10,  # AEST
        "Perth": 8,      # AWST
        "Auckland": 12,  # NZST
        "Australia": 10, # AEST (East coast)
        "New Zealand": 12, # NZST

        # South America
        "Sao Paulo": -3, # BRT
        "Rio de Janeiro": -3, # BRT
        "Buenos Aires": -3, # ART
        "Lima": -5,      # PET
        "Bogota": -5,    # COT
        "Brazil": -3,    # BRT
        "Argentina": -3, # ART
        "Peru": -5,      # PET
        "Colombia": -5,  # COT

        # Africa
        "Cairo": 2,      # EET
        "Lagos": 1,      # WAT
        "Johannesburg": 2, # SAST
        "Nairobi": 3,    # EAT
        "Cape Town": 2,  # SAST
        "Egypt": 2,      # EET
        "Nigeria": 1,    # WAT
        "South Africa": 2, # SAST
        "Kenya": 3,      # EAT
    }

    # Try to find the location in our database (case-insensitive)
    location_key = None

    # First try exact match (case-insensitive)
    for key in time_offsets.keys():
        if key.lower() == location.lower():
            location_key = key
            break

    # If no exact match, try partial match
    if not location_key:
        for key in time_offsets.keys():
            if location.lower() in key.lower() or key.lower() in location.lower():
                location_key = key
                break

    # Default to UTC if location not found
    offset = time_offsets.get(location_key, 0) if location_key else 0

    # Get timezone name based on offset
    timezone_names = {
        -8: "PST (Pacific Standard Time)",
        -7: "PDT (Pacific Daylight Time)",
        -6: "CST (Central Standard Time)",
        -5: "EST (Eastern Standard Time)",
        -4: "EDT (Eastern Daylight Time)",
        0: "GMT (Greenwich Mean Time)",
        1: "BST/CET (British Summer Time/Central European Time)",
        2: "CEST (Central European Summer Time)",
        3: "MSK (Moscow Standard Time)",
        4: "GST (Gulf Standard Time)",
        5: "PKT (Pakistan Standard Time)",
        5.5: "IST (Indian Standard Time)",
        8: "CST/SGT (China Standard Time/Singapore Time)",
        9: "JST (Japan Standard Time)",
        10: "AEST (Australian Eastern Standard Time)",
        12: "NZST (New Zealand Standard Time)"
    }

    timezone_name = timezone_names.get(offset, f"UTC{'+' if offset >= 0 else ''}{offset}")

    # Get current UTC time and apply offset
    utc_time = datetime.datetime.now(datetime.timezone.utc)  # Using timezone-aware datetime
    local_time = utc_time + datetime.timedelta(hours=offset)

    # Format time according to preference
    if format == "12h":
        time_str = local_time.strftime("%I:%M %p")
    else:
        time_str = local_time.strftime("%H:%M")

    date_str = local_time.strftime("%A, %B %d, %Y")

    return {
        "location": location,
        "time": time_str,
        "date": date_str,
        "timezone": f"UTC{'+' if offset >= 0 else ''}{offset}",
        "format": format
    }

def get_real_time(location, format="24h"):
    """Get accurate time for a location using TimeZoneDB API and OpenCage Geocoding API"""
    timezonedb_api_key = os.environ.get('TIMEZONEDB_API_KEY')
    opencage_api_key = os.environ.get('OPENCAGE_API_KEY')

    # If no API keys are provided, fall back to simulated data
    if not timezonedb_api_key or timezonedb_api_key == "your_timezonedb_api_key":
        print("TimeZoneDB API key not provided")
        return {
            "note": "Using simulated data (TimeZoneDB API key not provided)",
            **get_current_time(location, format)
        }

    if not opencage_api_key or opencage_api_key == "9e9a3a7c9e8a4c0e9a3a7c9e8a4c0e9a":
        print("OpenCage API key not provided")
        return {
            "note": "Using simulated data (OpenCage API key not provided)",
            **get_current_time(location, format)
        }

    try:
        # First, get coordinates for the location using OpenCage Geocoding API
        print(f"Getting coordinates for '{location}' using OpenCage API")
        geocode_url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={opencage_api_key}&limit=1"
        geocode_response = requests.get(geocode_url, timeout=5)

        if geocode_response.status_code != 200:
            print(f"OpenCage API error: {geocode_response.status_code}")
            return {
                "note": f"Error getting coordinates: {geocode_response.status_code}",
                **get_current_time(location, format)
            }

        geocode_data = geocode_response.json()

        # Check if we got any results
        if len(geocode_data.get('results', [])) == 0:
            print(f"No geocoding results found for '{location}'")
            return {
                "note": f"No geocoding results found for '{location}'",
                **get_current_time(location, format)
            }

        # Get the first result
        result = geocode_data['results'][0]
        lat = result['geometry']['lat']
        lng = result['geometry']['lng']
        formatted_location = result['formatted']

        print(f"Found coordinates for '{location}': {lat}, {lng} ({formatted_location})")

        # Make API request to TimeZoneDB
        print(f"Getting timezone data for coordinates {lat}, {lng}")
        timezone_url = f"https://api.timezonedb.com/v2.1/get-time-zone?key={timezonedb_api_key}&format=json&by=position&lat={lat}&lng={lng}"
        response = requests.get(timezone_url, timeout=5)

        # Check if request was successful
        if response.status_code == 200:
            data = response.json()

            if data["status"] == "OK":
                # Parse the timestamp
                timestamp = data["timestamp"]
                local_time = datetime.datetime.fromtimestamp(timestamp)

                # Format time according to preference
                if format == "12h":
                    time_str = local_time.strftime("%I:%M %p")
                else:
                    time_str = local_time.strftime("%H:%M")

                date_str = local_time.strftime("%A, %B %d, %Y")

                # Calculate GMT offset in hours
                gmt_offset_hours = data["gmtOffset"] / 3600

                return {
                    "location": location,
                    "time": time_str,
                    "date": date_str,
                    "timezone": data["zoneName"],
                    "timezone_abbreviation": data["abbreviation"],
                    "gmt_offset": f"GMT{'+' if gmt_offset_hours >= 0 else ''}{gmt_offset_hours}",
                    "dst": "Yes" if data["dst"] == "1" else "No",
                    "format": format
                }
            else:
                # If API returned an error, fall back to simulated data
                return {
                    "error": f"TimeZoneDB API error: {data.get('message', 'Unknown error')}",
                    "note": "Using simulated data as fallback",
                    **get_current_time(location, format)
                }
        else:
            # If API request failed, return error and fall back to simulated data
            return {
                "error": f"Could not retrieve timezone data: {response.status_code}",
                "note": "Using simulated data as fallback",
                **get_current_time(location, format)
            }
    except Exception as e:
        # If any exception occurs, fall back to simulated data
        return {
            "error": f"Error accessing timezone API: {str(e)}",
            "note": "Using simulated data as fallback",
            **get_current_time(location, format)
        }

def search_wikipedia(query, limit=1):
    """Search Wikipedia for information on a topic using the live Wikipedia API"""
    try:
        logger.info(f"Searching Wikipedia for '{query}'")
        # Make API request to Wikipedia with expanded parameters for better results
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": min(limit, 5),  # Limit to 5 results max
            "srprop": "snippet|titlesnippet|sectiontitle|categorysnippet|size|wordcount|timestamp|redirecttitle",
            "srinfo": "totalhits|suggestion|rewrittenquery",
            "srqiprofile": "classic_noboostlinks",  # Better quality results
            "srwhat": "text"  # Search in article text
        }

        # Add retry logic for Wikipedia API
        max_retries = 3
        retry_delay = 1  # seconds

        for retry in range(max_retries):
            try:
                search_response = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params=search_params,
                    timeout=5
                )

                # Break if successful
                if search_response.status_code == 200:
                    break

                # Wait before retrying
                if retry < max_retries - 1:
                    logger.warning(f"Wikipedia search retry {retry+1}/{max_retries} after error: {search_response.status_code}")
                    time.sleep(retry_delay * (retry + 1))
            except Exception as e:
                logger.error(f"Wikipedia search request error (attempt {retry+1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    time.sleep(retry_delay * (retry + 1))

        # Check if all retries failed
        if search_response.status_code != 200:
            logger.error(f"Wikipedia search failed after {max_retries} attempts: {search_response.status_code}")
            return {
                "error": f"Wikipedia search failed: {search_response.status_code}",
                "query": query,
                "data_source": "Wikipedia API (real-time)",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        search_data = search_response.json()
        search_results = search_data.get("query", {}).get("search", [])

        # Check for search suggestions if no results found
        if not search_results and "searchinfo" in search_data.get("query", {}):
            searchinfo = search_data["query"]["searchinfo"]
            suggestion = searchinfo.get("suggestion", "")

            if suggestion:
                logger.info(f"No results for '{query}', but found suggestion: '{suggestion}'")
                # Try the search again with the suggestion
                return search_wikipedia(suggestion, limit)

        if not search_results:
            logger.info(f"No Wikipedia articles found for '{query}'")
            return {
                "query": query,
                "results_count": 0,
                "message": f"No Wikipedia articles found for '{query}'",
                "results": [],
                "data_source": "Wikipedia API (real-time)",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        # For each search result, get the page content
        results = []

        for result in search_results:
            page_title = result["title"]
            logger.info(f"Getting content for Wikipedia article: {page_title}")

            # Get page content with expanded properties for more comprehensive data
            content_params = {
                "action": "query",
                "format": "json",
                "prop": "extracts|info|pageimages|categories|coordinates|langlinks|description|pageviews|revisions",
                "exintro": True,  # Only get the intro section
                "explaintext": True,  # Get plain text
                "inprop": "url|displaytitle|watchers|visitingwatchers",  # Get the URL and display title
                "pithumbsize": 500,  # Get a thumbnail image
                "titles": page_title,
                "redirects": 1,  # Follow redirects
                "cllimit": 5,  # Limit to 5 categories
                "lllimit": 5,  # Limit to 5 language links
                "lllang": "fr|es|de|it|ja",  # Get links for these languages
                "rvprop": "timestamp|user|comment",  # Get revision info
                "rvlimit": 1,  # Only get the latest revision
                "pvipdays": 7  # Get pageviews for the last 7 days
            }

            # Add retry logic for content retrieval
            max_retries = 3
            retry_delay = 1  # seconds
            content_response = None

            for retry in range(max_retries):
                try:
                    content_response = requests.get(
                        "https://en.wikipedia.org/w/api.php",
                        params=content_params,
                        timeout=5
                    )

                    # Break if successful
                    if content_response.status_code == 200:
                        break

                    # Wait before retrying
                    if retry < max_retries - 1:
                        logger.warning(f"Wikipedia content retry {retry+1}/{max_retries} for '{page_title}' after error: {content_response.status_code}")
                        time.sleep(retry_delay * (retry + 1))
                except Exception as e:
                    logger.error(f"Wikipedia content request error for '{page_title}' (attempt {retry+1}/{max_retries}): {str(e)}")
                    if retry < max_retries - 1:
                        time.sleep(retry_delay * (retry + 1))

            # Skip this result if all retries failed
            if not content_response or content_response.status_code != 200:
                logger.error(f"Failed to get content for '{page_title}' after {max_retries} attempts")
                continue

            content_data = content_response.json()
            pages = content_data.get("query", {}).get("pages", {})

            # Get the first (and only) page
            page_id = next(iter(pages))
            page = pages[page_id]

            # Extract the content
            extract = page.get("extract", "No content available")
            url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}")
            description = page.get("description", "")

            # Get last modified info
            last_modified = None
            if "revisions" in page and len(page["revisions"]) > 0:
                last_modified = page["revisions"][0].get("timestamp", "")

            # Get thumbnail image if available
            thumbnail = None
            if "thumbnail" in page:
                thumbnail = page["thumbnail"]["source"]

                # Get categories if available
                categories = []
                if "categories" in page:
                    for category in page["categories"]:
                        cat_title = category.get("title", "")
                        if cat_title.startswith("Category:"):
                            categories.append(cat_title[9:])  # Remove "Category:" prefix

                # Get coordinates if available
                coordinates = None
                if "coordinates" in page:
                    coord = page["coordinates"][0]
                    coordinates = {
                        "lat": coord.get("lat"),
                        "lon": coord.get("lon"),
                        "globe": coord.get("globe", "earth")
                    }

                # Get language links if available
                language_links = {}
                if "langlinks" in page:
                    for lang in page["langlinks"]:
                        language_links[lang["lang"]] = {
                            "title": lang["*"],
                            "url": f"https://{lang['lang']}.wikipedia.org/wiki/{lang['*'].replace(' ', '_')}"
                        }

                # Add to results
                results.append({
                    "title": page_title,
                    "extract": extract,
                    "url": url,
                    "thumbnail": thumbnail,
                    "categories": categories[:5] if categories else [],
                    "coordinates": coordinates,
                    "language_links": language_links,
                    "word_count": result.get("wordcount", 0),
                    "last_modified": result.get("timestamp", "")
                })

        # Return the results with enhanced real-time data indicators
        return {
            "query": query,
            "results_count": len(results),
            "results": results,
            "data_source": "Wikipedia API (real-time data)",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "search_time": f"{datetime.datetime.now().strftime('%H:%M:%S')}",
            "api_version": "live"
        }
    except Exception as e:
        logger.error(f"Error searching Wikipedia: {str(e)}", exc_info=True)
        return {
            "error": f"Error searching Wikipedia: {str(e)}",
            "query": query,
            "data_source": "Wikipedia API (real-time data)",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "api_version": "live"
        }

def search_products(query, category=None, max_price=None, sort_by="popularity"):
    """Simulate searching for products"""
    # In a real implementation, this would query a product database
    products = [
        {"id": 1, "name": "Smartphone X", "price": 799.99, "category": "Electronics", "rating": 4.5},
        {"id": 2, "name": "Laptop Pro", "price": 1299.99, "category": "Electronics", "rating": 4.8},
        {"id": 3, "name": "Wireless Headphones", "price": 149.99, "category": "Electronics", "rating": 4.3},
        {"id": 4, "name": "Running Shoes", "price": 89.99, "category": "Sports", "rating": 4.2},
        {"id": 5, "name": "Coffee Maker", "price": 59.99, "category": "Kitchen", "rating": 4.0},
        {"id": 6, "name": "Fitness Tracker", "price": 79.99, "category": "Electronics", "rating": 4.1},
        {"id": 7, "name": "Backpack", "price": 49.99, "category": "Fashion", "rating": 4.4},
        {"id": 8, "name": "Smart Watch", "price": 199.99, "category": "Electronics", "rating": 4.6},
    ]

    # Filter by query (simple contains check)
    results = [p for p in products if query.lower() in p["name"].lower()]

    # Apply category filter if provided
    if category:
        results = [p for p in results if p["category"].lower() == category.lower()]

    # Apply max price filter if provided
    if max_price:
        results = [p for p in results if p["price"] <= float(max_price)]

    # Sort results
    if sort_by == "price_asc":
        results.sort(key=lambda p: p["price"])
    elif sort_by == "price_desc":
        results.sort(key=lambda p: p["price"], reverse=True)
    elif sort_by == "rating":
        results.sort(key=lambda p: p["rating"], reverse=True)
    # Default is popularity (no sorting needed as the list is already in popularity order)

    return {
        "query": query,
        "category": category,
        "max_price": max_price,
        "sort_by": sort_by,
        "count": len(results),
        "results": results
    }

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    logger.info("Received chat request")
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free')
    # Get user-specified temperature or use default
    temperature = data.get('temperature', None)
    # Get quick mode setting (default to True for faster responses)
    quick_mode = data.get('quick_mode', True)
    # Get conversation history
    messages = data.get('messages', [])

    # Handle message formatting based on the model
    if model == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free":
        # For DeepSeek R1, we need special formatting
        if not messages:
            # Create a new conversation with just the user prompt
            # Check if this might be a math problem
            math_keywords = ["calculate", "solve", "equation", "math", "arithmetic", "algebra",
                            "geometry", "calculus", "trigonometry", "computation", "formula",
                            "+", "-", "*", "/", "=", "^", "square root", "derivative", "integral"]

            is_math_problem = any(keyword in prompt.lower() for keyword in math_keywords)

            if is_math_problem:
                # Special formatting for math problems
                formatted_prompt = f"Please answer the following math problem. Start your response with <think> to show your reasoning process step by step, and put your final answer within \\boxed{{}}.\n\nProblem: {prompt}"
            else:
                # Standard formatting for other questions
                formatted_prompt = f"Please answer the following question. Start your response with <think> to show your reasoning process, then provide your final answer.\n\nQuestion: {prompt}"

            messages = [{'role': 'user', 'content': formatted_prompt}]
        else:
            # Format existing messages for DeepSeek R1
            # If this is a multi-turn conversation, we need to ensure proper formatting
            # We'll only format the last user message if it doesn't already have our formatting
            for i in range(len(messages)):
                if messages[i]['role'] == 'user':
                    content = messages[i]['content']
                    # Only format if it doesn't already contain our formatting
                    if not content.startswith("Please answer") and not "Start your response with <think>" in content:
                        # Check if this might be a math problem
                        math_keywords = ["calculate", "solve", "equation", "math", "arithmetic", "algebra",
                                        "geometry", "calculus", "trigonometry", "computation", "formula",
                                        "+", "-", "*", "/", "=", "^", "square root", "derivative", "integral"]

                        is_math_problem = any(keyword in content.lower() for keyword in math_keywords)

                        if is_math_problem:
                            # Special formatting for math problems
                            messages[i]['content'] = f"Please answer the following math problem. Start your response with <think> to show your reasoning process step by step, and put your final answer within \\boxed{{}}.\n\nProblem: {content}"
                        else:
                            # Standard formatting for other questions
                            messages[i]['content'] = f"Please answer the following question. Start your response with <think> to show your reasoning process, then provide your final answer.\n\nQuestion: {content}"
    else:
        # Default format for other models like Llama
        if not messages:
            messages = [{'role': 'user', 'content': prompt}]

    if not prompt and not messages:
        return jsonify({"error": "Prompt or messages are required"}), 400

    # Special case for popular queries
    if prompt:
        # Case for "how to use llama models" query
        if "how to use llama models" in prompt.lower():
            # Add a specific system message for this query
            if not any(msg.get('role') == 'system' for msg in messages):
                # Use an ultra-concise system message if quick mode is enabled
                if quick_mode:
                    system_message = {
                        'role': 'system',
                        'content': """You are an AI assistant optimized for PRECISION and BREVITY. For Llama models, provide ONLY the most essential information:

1. Keep under 100 tokens total
2. List ONLY these 3 methods:
   - Hugging Face
   - llama.cpp
   - Ollama
3. For each: name + install command + one-line description
4. No examples unless explicitly requested
5. No explanations or background
6. Use only basic bullet points
7. Focus on the MOST IMPORTANT information only

REMEMBER: Precision and brevity are the absolute priorities."""
                    }
                else:
                    system_message = {
                        'role': 'system',
                        'content': """You are a helpful AI assistant that provides COMPREHENSIVE and DETAILED responses. For this query about using Llama models, create a well-structured response:

1. Start with a clear introduction explaining what LLaMA (Large Language Model Meta AI) models are
2. Use emoji icons for each major section (�, 💻, 🧩, 🔬, etc.)
3. Format your response with clear markdown structure:
   - Use ## for main heading and ### for section headings
   - Include emoji icons in section headings
   - Use horizontal rules (---) between major sections
   - Create a summary table at the end comparing different methods
4. For each method of using LLaMA models:
   - Include a ✅ Requirements section
   - Show code examples in ```python or ```bash code blocks
   - Use blockquotes for tips and important notes
   - Use bullet points for features and capabilities
5. Include specific sections on:
   - Using with Hugging Face Transformers
   - Using with llama.cpp
   - Using with Ollama
   - Fine-tuning options
   - Web UI options
6. End with a question asking if they need help with a specific implementation

Make your response visually appealing with proper spacing, emojis, and clear organization. Use a friendly, helpful tone that makes complex concepts easy to understand."""
                    }
                messages.insert(0, system_message)

        # Case for "compare deepseek and llama" query
        elif "compare deepseek and llama" in prompt.lower():
            # Add a specific system message for this query
            if not any(msg.get('role') == 'system' for msg in messages):
                # Use an ultra-concise system message if quick mode is enabled
                if quick_mode:
                    system_message = {
                        'role': 'system',
                        'content': """You are an AI assistant optimized for EXTREME SPEED. Compare DeepSeek vs Llama:

1. Keep under 50 tokens total
2. List ONLY key differences:
   - Performance
   - Size
   - Use cases
3. Format as "Model: feature"
4. No explanations
5. No formatting
6. No introductions

REMEMBER: Speed is the absolute priority."""
                    }
                else:
                    system_message = {
                        'role': 'system',
                        'content': """You are a helpful AI assistant optimized for SPEED. For this comparison between DeepSeek and Llama models, create a VERY CONCISE response:

1. Keep your response under 150 tokens for maximum speed
2. Use a simple two-column format with bullet points
3. Focus ONLY on these key differences:
   - Performance
   - Size options
   - Best use cases
4. Skip all explanations, background, and theory
5. No tables, no horizontal rules, no extra formatting
6. Use emoji icons for key points (🔑, 💡, ⚠️, ✅)

Remember: The user values SPEED over comprehensiveness. Give them just the key differences between the models as quickly as possible."""
                    }
                messages.insert(0, system_message)

        # Case for weather queries
        elif any(term in prompt.lower() for term in ["weather", "temperature", "forecast", "rain", "sunny", "cloudy"]):
            # Add a specific system message for weather queries
            if not any(msg.get('role') == 'system' for msg in messages):
                system_message = {
                    'role': 'system',
                    'content': """You are a helpful AI assistant. For this weather-related query, you MUST:

1. ALWAYS use the get_real_weather function to fetch current weather information
2. NEVER use the get_weather function (which provides simulated data)
3. NEVER make up or guess weather information - only use real-time API data
4. Format your response in a simple, plain text format:
   - No markdown headings (no ## or ###)
   - No emoji
   - No special formatting
   - Just present the information in clear, concise sentences
   - Include temperature, conditions, humidity, wind, and other details
5. If the location is ambiguous or not specified, ask for clarification
6. Briefly mention you're providing real-time data from OpenWeatherMap API

IMPORTANT: Real-time accuracy is the top priority. Always prioritize live data over cached information."""
                }
                messages.insert(0, system_message)

        # Case for time-related queries
        elif any(term in prompt.lower() for term in ["time", "clock", "hour", "timezone", "what time"]):
            # Add a specific system message for time queries
            if not any(msg.get('role') == 'system' for msg in messages):
                system_message = {
                    'role': 'system',
                    'content': """You are a helpful AI assistant. For this time-related query, you MUST:

1. ALWAYS use the get_real_time function to fetch current time information
2. NEVER use the get_current_time function (which provides simulated data)
3. NEVER make up or guess time information - only use real-time API data
4. Format your response in a simple, plain text format:
   - No markdown headings (no ## or ###)
   - No emoji
   - No special formatting
   - Just present the information in clear, concise sentences
   - Include the time, date, timezone, and DST information
5. If the location is ambiguous or not specified, ask for clarification
6. Briefly mention you're providing real-time data from TimeZoneDB API

IMPORTANT: Real-time accuracy is the top priority. Always prioritize live data over cached information."""
                }
                messages.insert(0, system_message)

        # Case for Wikipedia search queries
        elif any(term in prompt.lower() for term in ["wikipedia", "wiki", "article", "encyclopedia", "information about", "tell me about", "what is", "who is"]):
            # Add a specific system message for Wikipedia queries
            if not any(msg.get('role') == 'system' for msg in messages):
                system_message = {
                    'role': 'system',
                    'content': """You are a helpful AI assistant. For this information query, you MUST:

1. ALWAYS use the search_wikipedia function to fetch real-time information
2. NEVER make up or guess information - only use real-time API data
3. Format your response in a simple, plain text format:
   - No markdown headings (no ## or ###)
   - No emoji
   - No special formatting
   - Just present the information in clear, concise sentences
   - Include relevant details from the Wikipedia article
   - Cite Wikipedia as your source
4. If the query is ambiguous or too broad, ask for clarification
5. Briefly mention you're providing real-time data from Wikipedia API

IMPORTANT: Real-time accuracy is the top priority. Always prioritize live data over cached information."""
                }
                messages.insert(0, system_message)

        # Case for product search queries
        elif any(term in prompt.lower() for term in ["find", "search for", "looking for", "buy", "purchase", "product", "headphones", "laptop", "phone"]):
            # Add a specific system message for product search queries
            if not any(msg.get('role') == 'system' for msg in messages):
                system_message = {
                    'role': 'system',
                    'content': """You are a helpful AI assistant. For this product search query, you should:

1. Recognize that this is a product search query
2. Use the search_products function to find relevant products
3. Format your response with a clear structure:
   - Start with a ## heading about the search results
   - Present the product data in a visually appealing way
   - Use a table format for comparing multiple products
   - Include prices, ratings, and other relevant details
4. If the search query is ambiguous or too broad, ask for clarification

Remember to use the function calling capability rather than making up product information."""
                }
                messages.insert(0, system_message)

        # Case for greetings and casual conversation
        elif any(term in prompt.lower() for term in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "how are you", "what's up", "nice to meet you"]):
            # Add a specific system message for greetings
            if not any(msg.get('role') == 'system' for msg in messages):
                system_message = {
                    'role': 'system',
                    'content': """You are a helpful AI assistant. This appears to be a greeting or casual conversation:

1. Respond in a friendly, conversational manner
2. DO NOT use any function calls for simple greetings
3. Keep your response brief and welcoming
4. You can ask how you can help the user today
5. Format your response with a clear, friendly tone

Remember: Simple greetings don't require API calls or function execution."""
                }
                messages.insert(0, system_message)

        # Default case for all other queries
        else:
            # Add a general system message for proper formatting
            if not any(msg.get('role') == 'system' for msg in messages):
                # Use an ultra-concise system message if quick mode is enabled
                if quick_mode:
                    system_message = {
                        'role': 'system',
                        'content': """You are an AI assistant optimized for PRECISION and BREVITY. Your responses must be ULTRA CONCISE:

1. Keep responses under 100 tokens - extremely brief
2. Use only bullet points - no paragraphs ever
3. One idea per bullet - maximum clarity
4. Skip ALL pleasantries and explanations
5. No introductions or conclusions
6. No examples unless explicitly requested
7. No formatting except basic bullets
8. Focus on the MOST IMPORTANT information only

REMEMBER: Precision and brevity are the absolute priorities. Answer in the fewest possible words while still providing the most critical information."""
                    }
                else:
                    system_message = {
                        'role': 'system',
                        'content': """You are a helpful AI assistant that provides COMPREHENSIVE and DETAILED responses. Format your responses using markdown for readability:

1. Use up to 500 tokens for thorough, detailed answers
2. Start with a clear ## heading that summarizes the topic
3. Use ### subheadings with relevant emoji icons to organize different sections
4. Use bullet points or numbered lists for multiple items
5. Use **bold** for emphasis on important points
6. Break your response into clear paragraphs with proper spacing
7. Use code blocks when showing code or commands
8. Use tables for comparing multiple items when appropriate
9. End with a brief conclusion or summary paragraph

FORMATTING GUIDELINES:
- Use emoji icons in headings and key points (e.g., 🔑, 💡, ⚠️, ✅, etc.)
- Create visually appealing tables with clear headers
- Use blockquotes for important notes or tips
- Format code examples with proper syntax highlighting

IMPORTANT PRINCIPLES:
- Make complex concepts seem simple, not simple concepts seem complex
- Use clear, straightforward language rather than unnecessarily technical jargon
- Prioritize clarity and understanding over appearing intellectual
- Remember that true intelligence makes difficult ideas accessible
- Choose words and examples that illuminate rather than obscure

Remember: The user values COMPREHENSIVE and DETAILED information. Give them a thorough answer that covers all important aspects of their question."""
                    }
                messages.insert(0, system_message)

    # Validate the model
    if model not in VALID_MODELS:
        # Log the invalid model attempt
        print(f"Warning: Invalid model requested: {model}. Falling back to default model.")
        # Fallback to default model if invalid
        model = 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'

    # Get model settings
    model_settings = VALID_MODELS[model]

    # Use user-specified temperature if provided, otherwise use model default
    if temperature is None:
        temperature = model_settings["temperature"]
    else:
        try:
            temperature = float(temperature)
            # Ensure temperature is within valid range
            temperature = max(0.0, min(1.0, temperature))
        except ValueError:
            # If temperature is not a valid float, use model default
            temperature = model_settings["temperature"]

    try:
        logger.debug(f"Processing chat with model: {model}, quick_mode: {quick_mode}")
        # Prepare API request
        api_request = {
            'model': model,
            'messages': messages,
            'max_tokens': model_settings["max_tokens_quick"] if quick_mode else model_settings["max_tokens"],  # Use specific token limits for each mode
            'temperature': temperature,
            'top_p': model_settings["top_p"]
        }

        # Add functions if the model supports them
        if model_settings.get("supports_functions", False):
            # Convert functions to tools format for Together API
            tools = [
                {
                    "type": "function",
                    "function": func_def
                }
                for func_def in AVAILABLE_FUNCTIONS.values()
            ]
            api_request['tools'] = tools

        # System message is already added based on query type, no need to add another one

        logger.debug(f"Sending API request to Together API")
        # Call Together API
        response = requests.post(
            'https://api.together.xyz/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {API_KEY}'
            },
            json=api_request,
            timeout=8 if quick_mode else 15  # Ultra-aggressive timeout in quick mode
        )

        # Check if the API returned an error
        if response.status_code != 200:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', 'Unknown API error')
            logger.error(f"Together API error: {error_message}")
            return jsonify({"error": f"Together API error: {error_message}"}), response.status_code

        # Get the response JSON
        response_json = response.json()

        # Check if the model wants to call a function
        if 'choices' in response_json and len(response_json['choices']) > 0:
            message = response_json['choices'][0]['message']

            # Initialize variables for function handling
            function_call = None
            function_name = None
            tool_call = None
            function_response = None

            # Handle function calling (support both function_call and tool_calls)
            if message.get('function_call'):
                # Legacy format
                function_call = message['function_call']
                function_name = function_call.get('name')
                logger.info(f"Detected function call: {function_name}")
            elif message.get('tool_calls') and len(message['tool_calls']) > 0:
                # New format with tool_calls
                tool_call = message['tool_calls'][0]  # Get the first tool call
                if tool_call.get('type') == 'function':
                    function_call = tool_call['function']
                    function_name = function_call.get('name')
                    logger.info(f"Detected tool call: {function_name}")

            # Only proceed if we have a valid function call
            if function_call and function_name:
                try:
                    logger.info(f"Processing function: {function_name}")
                    # Parse function arguments
                    arguments = json.loads(function_call.get('arguments', '{}'))
                    logger.debug(f"Arguments: {arguments}")

                    # Execute the function
                    function_response = execute_function(function_name, arguments)
                    logger.debug(f"Function response: {function_response}")

                    # Add the function call and response to messages
                    if tool_call:
                        # New format with tool_calls
                        messages.append({
                            'role': 'assistant',
                            'content': None,
                            'tool_calls': message['tool_calls']
                        })

                        messages.append({
                            'role': 'tool',
                            'tool_call_id': tool_call.get('id', ''),
                            'name': function_name,
                            'content': json.dumps(function_response)
                        })
                    else:
                        # Legacy format
                        messages.append({
                            'role': 'assistant',
                            'content': None,
                            'function_call': function_call
                        })

                        messages.append({
                            'role': 'function',
                            'name': function_name,
                            'content': json.dumps(function_response)
                        })

                    # Prepare the second API request
                    second_api_request = {
                        'model': model,
                        'messages': messages,
                        'max_tokens': model_settings["max_tokens_quick"] if quick_mode else model_settings["max_tokens"],  # Use specific token limits for each mode
                        'temperature': temperature,
                        'top_p': model_settings["top_p"]
                    }

                    # Add tools if the model supports them
                    if model_settings.get("supports_functions", False):
                        # Convert functions to tools format for Together API
                        tools = [
                            {
                                "type": "function",
                                "function": func_def
                            }
                            for func_def in AVAILABLE_FUNCTIONS.values()
                        ]
                        second_api_request['tools'] = tools

                    # System message is already added based on query type, no need to add another one
                    # Just make sure the messages in the request are up to date
                    second_api_request['messages'] = messages

                    logger.info("Sending second API request")
                    # Handle rate limiting with retries
                    max_retries = 3
                    retry_delay = 2  # seconds

                    for retry in range(max_retries):
                        try:
                            logger.info(f"Sending second API request (attempt {retry+1}/{max_retries})")
                            # Call the API again with the function result
                            second_response = requests.post(
                                'https://api.together.xyz/v1/chat/completions',
                                headers={
                                    'Content-Type': 'application/json',
                                    'Authorization': f'Bearer {API_KEY}'
                                },
                                json=second_api_request,
                                timeout=8 if quick_mode else 15  # Ultra-aggressive timeout in quick mode
                            )

                            if second_response.status_code == 200:
                                logger.info("Second API call successful")
                                response_json = second_response.json()

                                # Store function response for later use
                                response_json['_function_response'] = function_response
                                response_json['_function_name'] = function_name
                                break  # Success, exit the retry loop
                            elif second_response.status_code == 429:
                                # Rate limit hit, wait and retry
                                if retry < max_retries - 1:  # Don't wait on the last attempt
                                    wait_time = retry_delay * (retry + 1)  # Exponential backoff
                                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry")
                                    time.sleep(wait_time)
                                else:
                                    # Last retry failed
                                    logger.error(f"Rate limit persists after {max_retries} retries")
                                    # Instead of failing, return a direct response with the function result
                                    return jsonify({
                                        "choices": [{
                                            "message": {
                                                "role": "assistant",
                                                "content": f"I found the information you requested about {function_name}:\n\n" +
                                                          json.dumps(function_response, indent=2)
                                            }
                                        }]
                                    })
                            else:
                                # Other error
                                logger.error(f"Second API call failed: {second_response.status_code}")
                                if retry < max_retries - 1:  # Don't wait on the last attempt
                                    wait_time = retry_delay * (retry + 1)
                                    logger.info(f"Waiting {wait_time} seconds before retry")
                                    time.sleep(wait_time)
                                else:
                                    # Last retry failed with non-rate-limit error
                                    error_data = second_response.json() if second_response.text else {"message": "No response body"}
                                    # Instead of failing, return a direct response with the function result
                                    return jsonify({
                                        "choices": [{
                                            "message": {
                                                "role": "assistant",
                                                "content": f"I found the information you requested about {function_name}:\n\n" +
                                                          json.dumps(function_response, indent=2)
                                            }
                                        }]
                                    })
                        except Exception as e:
                            logger.error(f"Error during API call: {str(e)}", exc_info=True)
                            if retry < max_retries - 1:
                                wait_time = retry_delay * (retry + 1)
                                logger.info(f"Waiting {wait_time} seconds before retry")
                                time.sleep(wait_time)
                            else:
                                # Last retry failed with exception
                                # Return a direct response with the function result
                                return jsonify({
                                    "choices": [{
                                        "message": {
                                            "role": "assistant",
                                            "content": f"I found the information you requested about {function_name}:\n\n" +
                                                      json.dumps(function_response, indent=2)
                                        }
                                    }]
                                })

                except Exception as e:
                    logger.error(f"Function execution error: {str(e)}", exc_info=True)
                    # If function execution fails, return the error
                    return jsonify({
                        "error": f"Function execution failed: {str(e)}",
                        "function_call": function_call
                    }), 500

            # Convert content to HTML using Markdown if it exists
            if 'content' in message and message['content']:
                content = message['content']
                # Convert markdown to HTML with minimal extensions for maximum speed
                html_content = markdown.markdown(
                    content,
                    extensions=[
                        'extra',  # Basic markdown features
                        'codehilite'  # Just basic code highlighting
                    ],
                    extension_configs={
                        'codehilite': {
                            'linenums': False,
                            'css_class': 'highlight',
                            'noclasses': True  # Use inline styles for faster rendering
                        }
                    }
                )
                # Update the response with HTML content
                response_json['choices'][0]['message']['content'] = html_content

                # Add function call info to the response if applicable
                if '_function_response' in response_json and '_function_name' in response_json:
                    # Use the stored function response from earlier
                    response_json['function_executed'] = {
                        'name': response_json['_function_name'],
                        'result': response_json['_function_response']
                    }
                    # Remove the temporary storage keys
                    del response_json['_function_response']
                    del response_json['_function_name']
                elif message.get('function_call'):
                    response_json['function_executed'] = {
                        'name': message['function_call']['name'],
                        'result': function_response
                    }
                elif message.get('tool_calls') and len(message['tool_calls']) > 0:
                    tool_call = message['tool_calls'][0]
                    if tool_call.get('type') == 'function':
                        response_json['function_executed'] = {
                            'name': tool_call['function']['name'],
                            'result': function_response
                        }

        logger.info("Successfully processed chat request")
        return jsonify(response_json)
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500



# For Vercel deployment
# This is the entry point that Vercel looks for
def vercel_app():
    return app

if __name__ == '__main__':
    # Use environment variable for debug mode, default to False for security
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")
    port = int(os.getenv("PORT", 5001))

    if debug_mode:
        logger.warning("Running in DEBUG mode - not recommended for production")
    else:
        logger.info("Running in PRODUCTION mode")

    app.run(debug=debug_mode, port=port, host='0.0.0.0')
