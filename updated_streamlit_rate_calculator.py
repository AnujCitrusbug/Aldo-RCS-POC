import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

load_dotenv()

# Configure page
st.set_page_config(
    page_title="Transport Rate Calculator",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .info-card {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


class RateCalculator:
    """Streamlit-optimized version of the Rate Calculator with Excel distance integration"""

    def __init__(self, metro_definitions, zipcode_assignment, transcar_data, competitor_data, sample_routes_with_distance):
        self.metro_definitions = metro_definitions.copy()
        self.zipcode_assignment = zipcode_assignment.copy()
        self.transcar_data = transcar_data.copy()
        self.competitor_data = competitor_data.copy()
        self.sample_routes_with_distance = sample_routes_with_distance.copy()

        # Convert numeric columns for calculations
        self._prepare_data()

        # Configuration constants
        self.SHORT_DISTANCE_THRESHOLD = 100
        self.SHORT_DISTANCE_FLAT_RATE = 200
        self.FALLBACK_RATE_PER_MILE = 1.50
        self.FALLBACK_SURCHARGE_MULTIPLIER = 1.25
        self.TRANSCAR_WEIGHT = 0.7
        self.COMPETITOR_WEIGHT = 0.3
        
        # Enhanced fallback configuration
        self.NEARBY_METRO_THRESHOLD = 200  # miles
        self.ADMIN_BASE_PER_MILE_RATE = 1.25
        self.ADMIN_BASE_RATE_SURCHARGE = 0.25  # 25% surcharge
        
        # Initialize OpenAI client
        self.openai_client = None
        self._initialize_openai()

    def _initialize_openai(self):
        """Initialize OpenAI client with API key from .env file or Streamlit secrets."""
        try:
            # Try to get API key from Streamlit secrets first
            if hasattr(st, 'secrets') and 'openai' in st.secrets and 'api_key' in st.secrets.openai:
                api_key = st.secrets.openai.api_key
            else:
                # Fallback to environment variable from .env file
                api_key = os.getenv('OPEN_AI_KEY')
            
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                st.warning("⚠️ OpenAI API key not found. Enhanced fallback features will be limited.")
        except Exception as e:
            st.warning(f"⚠️ Failed to initialize OpenAI client: {str(e)}")

    def _prepare_data(self):
        """Convert string columns to appropriate numeric types for calculations."""
        # Metro definitions
        numeric_cols = ['Latitude', 'Longitude', 'radius_miles', 'Population']
        for col in numeric_cols:
            if col in self.metro_definitions.columns:
                self.metro_definitions[col] = pd.to_numeric(self.metro_definitions[col], errors='coerce')

        # Zipcode assignment
        self.zipcode_assignment['Latitude'] = pd.to_numeric(self.zipcode_assignment['Latitude'], errors='coerce')
        self.zipcode_assignment['Longitude'] = pd.to_numeric(self.zipcode_assignment['Longitude'], errors='coerce')
        self.zipcode_assignment['Zip_Code'] = pd.to_numeric(self.zipcode_assignment['Zip_Code'], errors='coerce')

        # Sample routes with distance - handle if DISTANCE column exists
        self.sample_routes_with_distance['ORIGIN ZIPCODE'] = pd.to_numeric(self.sample_routes_with_distance['ORIGIN ZIPCODE'], errors='coerce')
        self.sample_routes_with_distance['DEST ZIPCODE'] = pd.to_numeric(self.sample_routes_with_distance['DEST ZIPCODE'], errors='coerce')

        # Check if DISTANCE column exists, if not, add it with default values
        if 'DISTANCE' not in self.sample_routes_with_distance.columns:
            # Add default distances for the known routes
            default_distances = [305, 302, 511]  # Addison-Galveston, Denton-Baytown, Galveston-BigSpring
            self.sample_routes_with_distance['DISTANCE'] = default_distances[:len(self.sample_routes_with_distance)]

        self.sample_routes_with_distance['DISTANCE'] = pd.to_numeric(self.sample_routes_with_distance['DISTANCE'], errors='coerce')

        # Transcar data
        if 'Distance' in self.transcar_data.columns:
            self.transcar_data['Distance'] = pd.to_numeric(self.transcar_data['Distance'], errors='coerce')
        if 'Total Price per Mile' in self.transcar_data.columns:
            self.transcar_data['Total Price per Mile'] = pd.to_numeric(self.transcar_data['Total Price per Mile'], errors='coerce')
        if 'Pickup ZIP' in self.transcar_data.columns:
            self.transcar_data['Pickup ZIP'] = pd.to_numeric(self.transcar_data['Pickup ZIP'], errors='coerce')
        if 'Delivery ZIP' in self.transcar_data.columns:
            self.transcar_data['Delivery ZIP'] = pd.to_numeric(self.transcar_data['Delivery ZIP'], errors='coerce')

        # Competitor data
        if 'Distance' in self.competitor_data.columns:
            self.competitor_data['Distance'] = pd.to_numeric(self.competitor_data['Distance'], errors='coerce')
        if 'competitor price per Mile' in self.competitor_data.columns:
            self.competitor_data['competitor price per Mile'] = pd.to_numeric(self.competitor_data['competitor price per Mile'], errors='coerce')
        if 'Origin Zipcode' in self.competitor_data.columns:
            self.competitor_data['Origin Zipcode'] = pd.to_numeric(self.competitor_data['Origin Zipcode'], errors='coerce')
        if 'Dest Zipcode' in self.competitor_data.columns:
            self.competitor_data['Dest Zipcode'] = pd.to_numeric(self.competitor_data['Dest Zipcode'], errors='coerce')

    def get_zipcode_coords(self, zipcode):
        """Get latitude and longitude coordinates for a given ZIP code."""
        try:
            zipcode_int = int(float(str(zipcode)))
            zip_row = self.zipcode_assignment[self.zipcode_assignment['Zip_Code'] == zipcode_int]
            if not zip_row.empty:
                return float(zip_row.iloc[0]['Latitude']), float(zip_row.iloc[0]['Longitude'])
        except (ValueError, TypeError):
            pass
        return None, None

    def assign_metro(self, zipcode):
        """Assign a ZIP code to its nearest metro area."""
        lat, lon = self.get_zipcode_coords(zipcode)
        if lat is None or lon is None:
            return None, False, 999

        zip_location = (lat, lon)
        closest_metro = None
        min_distance = float('inf')
        is_within_metro = False

        for idx, metro in self.metro_definitions.iterrows():
            try:
                metro_location = (float(metro['Latitude']), float(metro['Longitude']))
                distance = geodesic(zip_location, metro_location).miles

                if distance < min_distance:
                    min_distance = distance
                    closest_metro = metro['Metro_Area']
                    radius = float(metro['radius_miles']) if not pd.isna(metro['radius_miles']) else 20
                    is_within_metro = distance <= radius
            except (ValueError, TypeError):
                continue

        return closest_metro, is_within_metro, min_distance

    def get_excel_distance(self, origin_zip, dest_zip):
        """Get distance from Excel file for sample routes, return None if not found."""
        try:
            origin_zip_int = int(float(str(origin_zip)))
            dest_zip_int = int(float(str(dest_zip)))

            # Check if this route exists in sample routes (both directions)
            route_match = self.sample_routes_with_distance[
                ((self.sample_routes_with_distance['ORIGIN ZIPCODE'] == origin_zip_int) & 
                 (self.sample_routes_with_distance['DEST ZIPCODE'] == dest_zip_int)) |
                ((self.sample_routes_with_distance['ORIGIN ZIPCODE'] == dest_zip_int) & 
                 (self.sample_routes_with_distance['DEST ZIPCODE'] == origin_zip_int))
            ]

            if not route_match.empty:
                distance = float(route_match.iloc[0]['DISTANCE'])
                return distance if not pd.isna(distance) else None

        except (ValueError, TypeError):
            pass

        return None

    def calculate_distance_geodesic(self, origin_zip, dest_zip):
        """Calculate geodesic distance between two ZIP codes as fallback."""
        origin_lat, origin_lon = self.get_zipcode_coords(origin_zip)
        dest_lat, dest_lon = self.get_zipcode_coords(dest_zip)

        if None in [origin_lat, origin_lon, dest_lat, dest_lon]:
            return 0

        try:
            origin_location = (origin_lat, origin_lon)
            dest_location = (dest_lat, dest_lon)
            return geodesic(origin_location, dest_location).miles
        except:
            return 0

    def calculate_distance(self, origin_zip, dest_zip):
        """
        Calculate distance with priority:
        1. First try to get distance from Excel file (sample routes)
        2. If not found, calculate using geodesic coordinates
        """
        # Try to get distance from Excel first
        excel_distance = self.get_excel_distance(origin_zip, dest_zip)
        if excel_distance is not None:
            return excel_distance, "Excel Data"

        # Fallback to geodesic calculation
        geodesic_distance = self.calculate_distance_geodesic(origin_zip, dest_zip)
        return geodesic_distance, "Calculated"

    def lookup_historical_data(self, origin_metro, dest_metro):
        """Search for historical pricing data between metro pairs."""
        # Search Transcar data
        transcar_rates = []
        for idx, row in self.transcar_data.iterrows():
            try:
                pickup_metro, _, _ = self.assign_metro(row['Pickup ZIP'])
                delivery_metro, _, _ = self.assign_metro(row['Delivery ZIP'])

                if ((pickup_metro == origin_metro and delivery_metro == dest_metro) or
                    (pickup_metro == dest_metro and delivery_metro == origin_metro)):
                    rate = float(row['Total Price per Mile'])
                    if not pd.isna(rate) and rate > 0:
                        transcar_rates.append(rate)
            except (ValueError, TypeError, KeyError):
                continue

        # Search Competitor data
        competitor_rates = []
        for idx, row in self.competitor_data.iterrows():
            try:
                origin_metro_comp, _, _ = self.assign_metro(row['Origin Zipcode'])
                dest_metro_comp, _, _ = self.assign_metro(row['Dest Zipcode'])

                if ((origin_metro_comp == origin_metro and dest_metro_comp == dest_metro) or
                    (origin_metro_comp == dest_metro and dest_metro_comp == origin_metro)):
                    rate = float(row['competitor price per Mile'])
                    if not pd.isna(rate) and rate > 0:
                        competitor_rates.append(rate)
            except (ValueError, TypeError, KeyError):
                continue

        # Calculate weighted average
        transcar_avg = np.mean(transcar_rates) if transcar_rates else 0
        competitor_avg = np.mean(competitor_rates) if competitor_rates else 0

        if transcar_rates and competitor_rates:
            weighted_rate = (self.TRANSCAR_WEIGHT * transcar_avg) + (self.COMPETITOR_WEIGHT * competitor_avg)
        elif transcar_rates:
            weighted_rate = transcar_avg
        elif competitor_rates:
            weighted_rate = competitor_avg
        else:
            weighted_rate = None

        return weighted_rate, len(transcar_rates), len(competitor_rates)

    def get_nearby_metros(self, metro_name, threshold_miles=200):
        """Get metros within threshold distance of the given metro."""
        if metro_name is None:
            return []
        
        try:
            # Find the metro coordinates
            metro_row = self.metro_definitions[self.metro_definitions['Metro_Area'] == metro_name]
            if metro_row.empty:
                return []
            
            metro_lat = float(metro_row.iloc[0]['Latitude'])
            metro_lon = float(metro_row.iloc[0]['Longitude'])
            metro_location = (metro_lat, metro_lon)
            
            nearby_metros = []
            for idx, other_metro in self.metro_definitions.iterrows():
                if other_metro['Metro_Area'] == metro_name:
                    continue
                
                try:
                    other_lat = float(other_metro['Latitude'])
                    other_lon = float(other_metro['Longitude'])
                    other_location = (other_lat, other_lon)
                    
                    distance = geodesic(metro_location, other_location).miles
                    if distance <= threshold_miles:
                        nearby_metros.append({
                            'metro': other_metro['Metro_Area'],
                            'distance': distance
                        })
                except (ValueError, TypeError):
                    continue
            
            return sorted(nearby_metros, key=lambda x: x['distance'])
        except (ValueError, TypeError, KeyError):
            return []

    def find_similar_routes_with_openai(self, origin_metro, dest_metro):
        """Use OpenAI to find similar routes and determine if they're within 200 miles."""
        if not self.openai_client:
            return None, "OpenAI not available"
        
        try:
            # Get all available metros for context
            all_metros = self.metro_definitions['Metro_Area'].tolist()
            
            prompt = f"""
            You are a transportation logistics expert. I need to find similar routes for transport pricing.

            Available metro areas: {', '.join(all_metros)}

            Target route: {origin_metro} → {dest_metro}

            Please analyze and suggest similar routes that would be within 200 miles of either the origin or destination metro.
            Consider:
            1. Nearby Origin Metro → Destination Metro (within 200 miles of origin)
            2. Origin Metro → Nearby Destination Metro (within 200 miles of destination)
            3. Geographic proximity and transportation patterns

            Return your response as a JSON object with this structure:
            {{
                "similar_routes": [
                    {{
                        "origin_metro": "metro_name",
                        "dest_metro": "metro_name", 
                        "similarity_type": "nearby_origin" or "nearby_destination",
                        "confidence": 0.0-1.0,
                        "reasoning": "brief explanation"
                    }}
                ],
                "recommended_route": {{
                    "origin_metro": "metro_name",
                    "dest_metro": "metro_name",
                    "confidence": 0.0-1.0
                }}
            }}

            Only suggest routes that are actually within 200 miles of the original route endpoints.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            result = json.loads(response.choices[0].message.content)
            return result, "OpenAI Analysis"
            
        except Exception as e:
            return None, f"OpenAI Error: {str(e)}"

    def lookup_enhanced_fallback_data(self, origin_metro, dest_metro):
        """Enhanced fallback: Search for similar routes using OpenAI and nearby metros."""
        fallback_results = []
        
        # Method 1: Use OpenAI to find similar routes
        if self.openai_client:
            ai_result, ai_status = self.find_similar_routes_with_openai(origin_metro, dest_metro)
            if ai_result and 'similar_routes' in ai_result:
                for route in ai_result['similar_routes']:
                    similar_origin = route['origin_metro']
                    similar_dest = route['dest_metro']
                    confidence = route.get('confidence', 0.5)
                    
                    # Look up historical data for similar route
                    historical_rate, transcar_count, competitor_count = self.lookup_historical_data(similar_origin, similar_dest)
                    if historical_rate is not None:
                        fallback_results.append({
                            'rate': historical_rate,
                            'transcar_count': transcar_count,
                            'competitor_count': competitor_count,
                            'method': 'AI Similar Route',
                            'confidence': confidence,
                            'route': f"{similar_origin} → {similar_dest}",
                            'reasoning': route.get('reasoning', 'AI suggested similar route')
                        })

        
        # Method 2: Nearby metro search (within 200 miles)
        nearby_origin_metros = self.get_nearby_metros(origin_metro, self.NEARBY_METRO_THRESHOLD)
        nearby_dest_metros = self.get_nearby_metros(dest_metro, self.NEARBY_METRO_THRESHOLD)
        
        # Search for routes: Nearby Origin → Destination
        for nearby_metro in nearby_origin_metros:
            historical_rate, transcar_count, competitor_count = self.lookup_historical_data(nearby_metro['metro'], dest_metro)
            if historical_rate is not None:
                fallback_results.append({
                    'rate': historical_rate,
                    'transcar_count': transcar_count,
                    'competitor_count': competitor_count,
                    'method': 'Nearby Origin Metro',
                    'confidence': 0.8,
                    'route': f"{nearby_metro['metro']} → {dest_metro}",
                    'reasoning': f"Origin metro within {nearby_metro['distance']:.1f} miles"
                })
        
        # Search for routes: Origin → Nearby Destination
        for nearby_metro in nearby_dest_metros:
            historical_rate, transcar_count, competitor_count = self.lookup_historical_data(origin_metro, nearby_metro['metro'])
            if historical_rate is not None:
                fallback_results.append({
                    'rate': historical_rate,
                    'transcar_count': transcar_count,
                    'competitor_count': competitor_count,
                    'method': 'Nearby Destination Metro',
                    'confidence': 0.8,
                    'route': f"{origin_metro} → {nearby_metro['metro']}",
                    'reasoning': f"Destination metro within {nearby_metro['distance']:.1f} miles"
                })

        # Return the best match (highest confidence)
        if fallback_results:
            best_match = max(fallback_results, key=lambda x: x['confidence'])
            return best_match['rate'], best_match['transcar_count'], best_match['competitor_count'], best_match
        
        return None, 0, 0, None

    def calculate_surcharge(self, distance_to_metro):
        """Calculate surcharge based on distance from nearest metro area."""
        if distance_to_metro <= 25:
            return 50
        elif distance_to_metro <= 50:
            return 75
        elif distance_to_metro <= 100:
            return 100
        elif distance_to_metro <= 150:
            return 150
        elif distance_to_metro <= 200:
            return 200
        else:
            return 250

    def calculate_rate(self, origin_zip, dest_zip, vehicle_type="Car - Sedan"):
        """Main rate calculation method with Excel distance integration."""
        result = {
            'origin_zip': origin_zip,
            'dest_zip': dest_zip,
            'vehicle_type': vehicle_type
        }

        # Step 1: Calculate distance (Excel first, then geodesic)
        distance, distance_source = self.calculate_distance(origin_zip, dest_zip)
        result['distance'] = distance
        result['distance_source'] = distance_source

        if distance == 0:
            result['error'] = 'Unable to calculate distance - ZIP codes not found'
            result['final_rate'] = 0
            return result

        # Step 2: Check for short distance flat rate
        if distance < self.SHORT_DISTANCE_THRESHOLD:
            result['base_rate'] = self.SHORT_DISTANCE_FLAT_RATE
            result['calculation_method'] = 'Short Distance Flat Rate'
            result['per_mile_rate'] = self.SHORT_DISTANCE_FLAT_RATE / distance
            result['origin_surcharge'] = 0
            result['dest_surcharge'] = 0
            result['total_surcharge'] = 0
            result['final_rate'] = self.SHORT_DISTANCE_FLAT_RATE
            return result

        # Step 3: Metro area assignment
        origin_metro, origin_within, origin_distance = self.assign_metro(origin_zip)
        dest_metro, dest_within, dest_distance = self.assign_metro(dest_zip)

        result.update({
            'origin_metro': origin_metro,
            'origin_within_metro': origin_within,
            'origin_distance_to_metro': origin_distance,
            'dest_metro': dest_metro,
            'dest_within_metro': dest_within,
            'dest_distance_to_metro': dest_distance
        })

        # Step 4: Historical data lookup
        historical_rate, transcar_count, competitor_count = self.lookup_historical_data(origin_metro, dest_metro)
        result['historical_per_mile_rate'] = historical_rate
        result['transcar_data_count'] = transcar_count
        result['competitor_data_count'] = competitor_count

        # Step 5: Enhanced fallback mechanism
        if historical_rate is not None:
            result['base_rate'] = historical_rate * distance
            result['per_mile_rate'] = historical_rate
            result['calculation_method'] = 'Historical Data'
        else:
            # Enhanced fallback: Search for similar routes
            enhanced_rate, enhanced_transcar_count, enhanced_competitor_count, fallback_info = self.lookup_enhanced_fallback_data(origin_metro, dest_metro)
            
            if enhanced_rate is not None:
                result['base_rate'] = enhanced_rate * distance
                result['per_mile_rate'] = enhanced_rate
                result['calculation_method'] = f'Enhanced Fallback ({fallback_info["method"]})'
                result['fallback_info'] = fallback_info
                result['enhanced_transcar_count'] = enhanced_transcar_count
                result['enhanced_competitor_count'] = enhanced_competitor_count
            else:
                # Ultimate fallback: Admin Base Rate
                if not origin_within and not dest_within:
                    # Both non-metro: Admin Base Rate + 25% surcharge
                    admin_rate = self.ADMIN_BASE_PER_MILE_RATE * (1 + self.ADMIN_BASE_RATE_SURCHARGE)
                    result['calculation_method'] = 'Admin Base Rate (Non-Metro +25%)'
                else:
                    # At least one metro: Standard Admin Base Rate
                    admin_rate = self.ADMIN_BASE_PER_MILE_RATE
                    result['calculation_method'] = 'Admin Base Rate'
                
                result['base_rate'] = admin_rate * distance
                result['per_mile_rate'] = admin_rate

        # Step 6: Calculate surcharges
        origin_surcharge = 0 if origin_within else self.calculate_surcharge(origin_distance)
        dest_surcharge = 0 if dest_within else self.calculate_surcharge(dest_distance)

        result['origin_surcharge'] = origin_surcharge
        result['dest_surcharge'] = dest_surcharge
        result['total_surcharge'] = origin_surcharge + dest_surcharge

        # Step 7: Final rate calculation
        result['final_rate'] = result['base_rate'] + result['total_surcharge']

        return result


def clean_column_names(header_row):
    """Clean column names by replacing NaN values with unique column names"""
    clean_headers = []
    for i, val in enumerate(header_row):
        if pd.isna(val) or str(val).strip() == '' or str(val) == 'nan':
            clean_headers.append(f'Col_{i}')
        else:
            clean_headers.append(str(val))
    return clean_headers


@st.cache_data
def load_data():
    """Load and cache the POC data including sample routes with distances"""
    try:
        df_all = pd.read_excel("POC-Data-3-new.xlsx", sheet_name='Sheet1', header=None)

        # Extract sample routes WITH DISTANCES - using safe column name handling
        sample_routes = df_all.iloc[2:9, 0:7].copy()

        # Clean column names to avoid reindexing error
        header_row = sample_routes.iloc[0].tolist()
        clean_headers = clean_column_names(header_row)
        sample_routes.columns = clean_headers

        # Clean the data
        sample_routes = sample_routes.iloc[1:].dropna(how='all').reset_index(drop=True)

        # Extract metro definitions
        metro_definitions = df_all.iloc[12:75, 0:7].copy()
        header_row = metro_definitions.iloc[0].tolist()
        clean_headers = clean_column_names(header_row)
        metro_definitions.columns = clean_headers
        metro_definitions = metro_definitions.iloc[1:].dropna(how='all').reset_index(drop=True)

        # Extract zipcode assignment (header at row 78, data from row 79, ends before transcar at row 90)
        zipcode_assignment = df_all.iloc[78:90, 0:5].copy()
        header_row = zipcode_assignment.iloc[0].tolist()
        clean_headers = clean_column_names(header_row[:5])  # Only take first 5 columns
        zipcode_assignment.columns = clean_headers
        zipcode_assignment = zipcode_assignment.iloc[1:].dropna(how='all').reset_index(drop=True)

        # Extract transcar data (section title at row 90, header at row 91, data from row 92, ends before competitor at row 146)
        transcar_data = df_all.iloc[91:146, 0:11].copy()
        header_row = transcar_data.iloc[0].tolist()
        clean_headers = clean_column_names(header_row)
        transcar_data.columns = clean_headers
        transcar_data = transcar_data.iloc[1:].dropna(how='all').reset_index(drop=True)

        # Extract competitor data (section title at row 146, header at row 147, data from row 148+)
        competitor_data = df_all.iloc[147:, 0:11].copy()
        header_row = competitor_data.iloc[0].tolist()
        clean_headers = clean_column_names(header_row)
        competitor_data.columns = clean_headers
        competitor_data = competitor_data.iloc[1:].dropna(how='all').reset_index(drop=True)

        # Expand zipcode coverage
        additional_zipcodes = [
            {'Zip_Code': 77013, 'City': 'Houston', 'State': 'TX', 'Latitude': 29.7604, 'Longitude': -95.3698},
            {'Zip_Code': 77044, 'City': 'Houston', 'State': 'TX', 'Latitude': 29.7604, 'Longitude': -95.3698},
            {'Zip_Code': 77084, 'City': 'Houston', 'State': 'TX', 'Latitude': 29.7604, 'Longitude': -95.3698},
            {'Zip_Code': 76001, 'City': 'Arlington', 'State': 'TX', 'Latitude': 32.7357, 'Longitude': -97.1081},
            {'Zip_Code': 73301, 'City': 'Austin', 'State': 'TX', 'Latitude': 30.2672, 'Longitude': -97.7431},
            {'Zip_Code': 78401, 'City': 'Corpus Christi', 'State': 'TX', 'Latitude': 27.8006, 'Longitude': -97.3964},
            {'Zip_Code': 75201, 'City': 'Dallas', 'State': 'TX', 'Latitude': 32.7767, 'Longitude': -96.7970},
            {'Zip_Code': 75217, 'City': 'Dallas', 'State': 'TX', 'Latitude': 32.7767, 'Longitude': -96.7970},
            {'Zip_Code': 75236, 'City': 'Dallas', 'State': 'TX', 'Latitude': 32.7767, 'Longitude': -96.7970},
            {'Zip_Code': 79424, 'City': 'Lubbock', 'State': 'TX', 'Latitude': 33.5779, 'Longitude': -101.8552},
            {'Zip_Code': 76901, 'City': 'San Angelo', 'State': 'TX', 'Latitude': 31.4638, 'Longitude': -100.4370},
            {'Zip_Code': 76905, 'City': 'San Angelo', 'State': 'TX', 'Latitude': 31.4638, 'Longitude': -100.4370}
        ]

        zipcode_assignment = pd.concat([zipcode_assignment, pd.DataFrame(additional_zipcodes)], ignore_index=True)

        return metro_definitions, zipcode_assignment, transcar_data, competitor_data, sample_routes

    except FileNotFoundError:
        st.error("POC-Data-3-new.xlsx file not found. Please ensure it's in the same directory as this app.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None, None, None


def get_location_options(sample_routes):
    """Get unique locations from sample routes for dropdowns"""
    locations = []
    location_map = {}

    for idx, route in sample_routes.iterrows():
        # Origin location
        origin_display = f"{route['ORIGIN CITY']}, {route['ORIGIN STATE']} ({route['ORIGIN ZIPCODE']})"
        origin_zip = route['ORIGIN ZIPCODE']

        # Destination location
        dest_display = f"{route['DEST CITY']}, {route['DEST STATE']} ({route['DEST ZIPCODE']})"
        dest_zip = route['DEST ZIPCODE']

        if origin_display not in location_map:
            locations.append(origin_display)
            location_map[origin_display] = origin_zip

        if dest_display not in location_map:
            locations.append(dest_display)
            location_map[dest_display] = dest_zip

    return sorted(locations), location_map


def create_route_visualization(origin_coords, dest_coords, origin_name, dest_name):
    """Create a map visualization of the route"""
    if None in origin_coords or None in dest_coords:
        return None

    # Create map with route
    fig = go.Figure()

    # Add origin and destination markers
    fig.add_trace(go.Scattermapbox(
        lat=[origin_coords[0], dest_coords[0]],
        lon=[origin_coords[1], dest_coords[1]],
        mode='markers+text',
        marker=dict(size=15, color=['green', 'red']),
        text=[f'Origin: {origin_name}', f'Destination: {dest_name}'],
        textposition='top center',
        name='Locations'
    ))

    # Add route line
    fig.add_trace(go.Scattermapbox(
        lat=[origin_coords[0], dest_coords[0]],
        lon=[origin_coords[1], dest_coords[1]],
        mode='lines',
        line=dict(width=3, color='blue'),
        name='Route'
    ))

    # Set map layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=(origin_coords[0] + dest_coords[0]) / 2,
                lon=(origin_coords[1] + dest_coords[1]) / 2
            ),
            zoom=6
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )

    return fig


def main():
    """Main Streamlit application"""

    # Header
    st.title("🚛 Transport Rate Calculator")
    st.markdown("**POC Implementation** - Calculate transport rates based on historical data and metro assignments")

    # Load data
    with st.spinner("Loading POC data..."):
        metro_definitions, zipcode_assignment, transcar_data, competitor_data, sample_routes = load_data()

    if metro_definitions is None:
        st.stop()

    # Initialize calculator with sample routes including distance data
    calculator = RateCalculator(metro_definitions, zipcode_assignment, transcar_data, competitor_data, sample_routes)

    # Get location options
    locations, location_map = get_location_options(sample_routes)
    vehicle_types = ['Car - Sedan']

    # Sidebar for inputs
    st.sidebar.header("📍 Route Selection")

    # Location dropdowns
    origin_location = st.sidebar.selectbox(
        "Select Origin Location:",
        options=locations,
        key="origin"
    )

    dest_location = st.sidebar.selectbox(
        "Select Destination Location:",
        options=locations,
        key="destination"
    )

    # Vehicle type selection
    vehicle_type = st.sidebar.selectbox(
        "Select Vehicle Type:",
        options=vehicle_types,
        key="vehicle"
    )

    # Calculate button
    calculate_button = st.sidebar.button("🚀 Calculate Rate", type="primary", width='stretch')

    # Validation
    if origin_location == dest_location:
        st.sidebar.warning("⚠️ Origin and destination cannot be the same!")
        calculate_button = False

    # Main content area
    if calculate_button:
        # Get ZIP codes from location map
        origin_zip = location_map[origin_location]
        dest_zip = location_map[dest_location]

        # Calculate rate
        with st.spinner("Calculating transport rate..."):
            result = calculator.calculate_rate(origin_zip, dest_zip, vehicle_type)

        # Display results
        if 'error' in result:
            st.error(f"❌ {result['error']}")
        else:
            # Main metrics - REMOVED CONFIDENCE SCORE
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="💰 Final Rate",
                    value=f"${result['final_rate']:.2f}",
                    delta=f"${result['per_mile_rate']:.3f}/mile"
                )

            with col2:
                st.metric(
                    label="📏 Distance",
                    value=f"{result['distance']:.1f} mi",
                    delta=f"Source: {result['distance_source']}"
                )

            with col3:
                st.metric(
                    label="🏙️ Metro Coverage",
                    value="Full" if result.get('origin_within_metro') and result.get('dest_within_metro') else "Partial",
                    delta=f"${result['total_surcharge']:.0f} surcharge"
                )

            st.divider()

            # Detailed breakdown in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Rate Breakdown", "🗺️ Route Map", "📈 Historical Data", "⚙️ Calculation Details"])

            with tab1:
                # Rate breakdown chart
                breakdown_data = {
                    'Component': ['Base Rate', 'Origin Surcharge', 'Destination Surcharge'],
                    'Amount': [result['base_rate'], result.get('origin_surcharge', 0), result.get('dest_surcharge', 0)]
                }

                fig = px.bar(
                    breakdown_data,
                    x='Component',
                    y='Amount',
                    title='Rate Breakdown by Component',
                    color='Component',
                    text='Amount'
                )
                fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, width=True)

                # Breakdown table
                breakdown_df = pd.DataFrame([
                    ['Base Rate', f"${result['base_rate']:.2f}", f"${result['per_mile_rate']:.3f}/mile × {result['distance']:.1f} miles"],
                    ['Origin Surcharge', f"${result.get('origin_surcharge', 0):.2f}", 
                     f"Outside metro: {result.get('origin_distance_to_metro', 0):.1f} miles" if result.get('origin_surcharge', 0) > 0 else "Within metro area"],
                    ['Destination Surcharge', f"${result.get('dest_surcharge', 0):.2f}",
                     f"Outside metro: {result.get('dest_distance_to_metro', 0):.1f} miles" if result.get('dest_surcharge', 0) > 0 else "Within metro area"],
                    ['**Total Rate**', f"**${result['final_rate']:.2f}**", f"**Final transport rate**"]
                ], columns=['Component', 'Amount', 'Details'])

                st.dataframe(breakdown_df, width='stretch', hide_index=True)

                # Distance source information
                if result['distance_source'] == 'Excel Data':
                    st.success("✅ Distance obtained from Excel file (predefined route)")
                else:
                    st.info("ℹ️ Distance calculated using geodesic coordinates")

            with tab2:
                # Route visualization
                origin_coords = calculator.get_zipcode_coords(origin_zip)
                dest_coords = calculator.get_zipcode_coords(dest_zip)

                if None not in origin_coords and None not in dest_coords:
                    fig = create_route_visualization(origin_coords, dest_coords, origin_location, dest_location)
                    if fig:
                        st.plotly_chart(fig, width='stretch')

                    # Route summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        **📍 Origin Details**
                        - Location: {origin_location}
                        - Metro Area: {result.get('origin_metro', 'Unknown')}
                        - Within Metro: {'✅ Yes' if result.get('origin_within_metro') else '❌ No'}
                        - Distance to Metro: {result.get('origin_distance_to_metro', 0):.1f} miles
                        """)

                    with col2:
                        st.markdown(f"""
                        **📍 Destination Details**
                        - Location: {dest_location}
                        - Metro Area: {result.get('dest_metro', 'Unknown')}
                        - Within Metro: {'✅ Yes' if result.get('dest_within_metro') else '❌ No'}
                        - Distance to Metro: {result.get('dest_distance_to_metro', 0):.1f} miles
                        """)

                    # Distance source details
                    st.markdown(f"""
                    **📏 Distance Details**
                    - Total Distance: {result['distance']:.1f} miles
                    - Distance Source: {result['distance_source']}
                    - {'Exact distance from Excel file' if result['distance_source'] == 'Excel Data' else 'Calculated using geodesic coordinates'}
                    """)
                else:
                    st.warning("Unable to display map - coordinate data not available")

            with tab3:
                # Historical data analysis
                transcar_count = result.get('transcar_data_count', 0)
                competitor_count = result.get('competitor_data_count', 0)
                enhanced_transcar_count = result.get('enhanced_transcar_count', 0)
                enhanced_competitor_count = result.get('enhanced_competitor_count', 0)
                fallback_info = result.get('fallback_info')

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("📊 Transcar Records", transcar_count + enhanced_transcar_count, "70% weight")

                with col2:
                    st.metric("🏢 Competitor Records", competitor_count + enhanced_competitor_count, "30% weight")

                with col3:
                    total_records = transcar_count + competitor_count + enhanced_transcar_count + enhanced_competitor_count
                    st.metric("📈 Total Records", total_records, "Available data")

                if total_records > 0:
                    if result.get('calculation_method') == 'Historical Data':
                        st.success(f"✅ Direct historical data available! Using {transcar_count + competitor_count} records for calculation.")
                    else:
                        st.success(f"✅ Enhanced fallback data found! Using similar route data.")

                    if result.get('historical_per_mile_rate'):
                        st.info(f"📊 Calculated historical rate: **${result['historical_per_mile_rate']:.3f} per mile**")

                        # Data source breakdown
                        if transcar_count > 0 and competitor_count > 0:
                            st.write(f"Rate calculated using 70% Transcar + 30% Competitor weighting")
                        elif transcar_count > 0:
                            st.write(f"Rate based on {transcar_count} Transcar records (preferred source)")
                        else:
                            st.write(f"Rate based on {competitor_count} Competitor records")
                    
                    # Enhanced fallback information
                    if fallback_info:
                        st.subheader("Enhanced OpenAI Results")
                        st.info(f"""
                        **Method**: {fallback_info['method']}  
                        **Similar Route**: {fallback_info['route']}  
                        **Confidence**: {fallback_info['confidence']:.1%}  
                        **Reasoning**: {fallback_info['reasoning']}
                        """)
                        
                        if fallback_info['method'] == 'AI Similar Route':
                            st.success("🎯 OpenAI successfully identified a similar route!")
                        else:
                            st.info("📍 Found nearby metro within 200 miles")
                else:
                    st.warning(f"⚠️ No historical data available for this metro pair. Using Admin Base Rate of ${calculator.ADMIN_BASE_PER_MILE_RATE}/mile.")

            with tab4:
                # Detailed calculation information
                st.subheader("🔧 Algorithm Details")

                # Algorithm steps
                steps = [
                    ("1. Distance Calculation", f"{result['distance']:.2f} miles from {result['distance_source']}"),
                    ("2. Short Distance Check", f"{'Applied $200 flat rate' if result['distance'] < 100 else 'Standard calculation applied'}"),
                    ("3. Metro Assignment", f"Origin: {result.get('origin_metro', 'Unknown')}, Destination: {result.get('dest_metro', 'Unknown')}"),
                    ("4. Historical Data Lookup", f"{transcar_count + competitor_count} direct records found"),
                    ("5. Enhanced Fallback", f"{'Used' if result.get('fallback_info') else 'Not needed'} - {result['calculation_method']}"),
                    ("6. Base Rate Calculation", f"${result['base_rate']:.2f} using {result['calculation_method']}"),
                    ("7. Surcharge Application", f"${result['total_surcharge']:.2f} total surcharges"),
                    ("8. Final Rate", f"${result['final_rate']:.2f} (${result['per_mile_rate']:.3f}/mile)")
                ]

                for step, description in steps:
                    st.write(f"**{step}**: {description}")

                # Configuration display
                st.subheader("⚙️ Algorithm Configuration")
                config_data = {
                    'Parameter': [
                        'Short Distance Threshold',
                        'Short Distance Flat Rate',
                        'Admin Base Per-Mile Rate',
                        'Admin Base Rate Surcharge',
                        'Nearby Metro Threshold',
                        'Transcar Data Weight',
                        'Competitor Data Weight',
                        'Fallback Surcharge Multiplier'
                    ],
                    'Value': [
                        f"{calculator.SHORT_DISTANCE_THRESHOLD} miles",
                        f"${calculator.SHORT_DISTANCE_FLAT_RATE}",
                        f"${calculator.ADMIN_BASE_PER_MILE_RATE}",
                        f"{calculator.ADMIN_BASE_RATE_SURCHARGE*100}%",
                        f"{calculator.NEARBY_METRO_THRESHOLD} miles",
                        f"{calculator.TRANSCAR_WEIGHT*100}%",
                        f"{calculator.COMPETITOR_WEIGHT*100}%",
                        f"{calculator.FALLBACK_SURCHARGE_MULTIPLIER}x"
                    ]
                }
                st.dataframe(pd.DataFrame(config_data), width='stretch', hide_index=True)

                # Enhanced fallback mechanism details
                st.subheader("🤖 Enhanced Fallback Mechanism")
                st.write("""
                **Fallback Priority Order:**
                1. **Historical Data**: Direct metro-to-metro match
                2. **AI Similar Routes**: OpenAI analyzes similar routes within 200 miles
                3. **Nearby Metro Search**: Find metros within 200 miles of origin/destination
                4. **Admin Base Rate**: Ultimate fallback with configurable rates
                
                **Enhanced Features:**
                - **OpenAI Integration**: Intelligent route similarity detection
                - **200-Mile Radius**: Search for nearby metros within 200 miles
                - **Smart Weighting**: Confidence-based route selection
                - **Non-Metro Surcharge**: 25% surcharge for both non-metro locations
                """)

                # Distance calculation method details
                st.subheader("📏 Distance Calculation Method")
                st.write("""
                **Priority Order:**
                1. **Excel Data** (Preferred): Use exact distances from sample routes (305, 302, 511 miles)
                2. **Geodesic Calculation** (Fallback): Calculate distance using ZIP code coordinates

                This ensures accurate distances for predefined routes while maintaining flexibility for other calculations.
                """)

    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Transport Rate Calculator! 👋

        This application calculates transport rates using a sophisticated algorithm that considers:

        - 📏 **Distance calculations** using Excel data (priority) or geodesic coordinates
        - 🏙️ **Metro area assignments** with configurable radius
        - 📊 **Historical data integration** (70% Transcar + 30% Competitor)
        - 🤖 **AI-enhanced fallback** using OpenAI for intelligent route similarity detection
        - 🎯 **200-mile radius search** for nearby metros and similar routes
        - 💰 **Smart surcharge system** for outside-metro locations

        ### 🚀 How to Use:
        1. Select your **origin location** from the dropdown
        2. Select your **destination location** from the dropdown  
        3. Choose your **vehicle type**
        4. Click **"Calculate Rate"** to get your quote

        ### 📍 Available Locations:
        """)

        # Display available locations with distances
        st.subheader("📋 Sample Routes from POC Data")
        display_routes = sample_routes.copy()

        if len(display_routes) > 0:
            display_routes['Route'] = display_routes.apply(
                lambda row: f"{row['ORIGIN CITY']}, {row['ORIGIN STATE']} → {row['DEST CITY']}, {row['DEST STATE']}", 
                axis=1
            )

            # Check if DISTANCE column exists
            columns_to_show = ['Route', 'ORIGIN ZIPCODE', 'DEST ZIPCODE']
            column_renames = {'ORIGIN ZIPCODE': 'Origin ZIP', 'DEST ZIPCODE': 'Destination ZIP'}

            if 'DISTANCE' in display_routes.columns:
                columns_to_show.append('DISTANCE')
                column_renames['DISTANCE'] = 'Distance (miles)'

            st.dataframe(
                display_routes[columns_to_show].rename(columns=column_renames),
                width='stretch',
                hide_index=True
            )

        # Available locations list
        st.subheader("🗺️ All Available Locations:")
        for i, location in enumerate(locations, 1):
            st.write(f"{i}. {location}")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Transport Rate Calculator POC** • "
        f"Loaded {len(metro_definitions)} metro areas • "
        f"{len(zipcode_assignment)} ZIP codes • "
        f"{len(transcar_data)} Transcar records • "
        f"{len(competitor_data)} competitor records • "
        f"{len(sample_routes)} sample routes"
    )


if __name__ == "__main__":
    main()
