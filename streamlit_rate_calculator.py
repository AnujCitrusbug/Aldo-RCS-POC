import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    """Streamlit-optimized version of the Rate Calculator"""

    def __init__(self, metro_definitions, zipcode_assignment, transcar_data, competitor_data):
        self.metro_definitions = metro_definitions.copy()
        self.zipcode_assignment = zipcode_assignment.copy()
        self.transcar_data = transcar_data.copy()
        self.competitor_data = competitor_data.copy()

        # Convert numeric columns for calculations
        self._prepare_data()

        # Configuration constants
        self.SHORT_DISTANCE_THRESHOLD = 100
        self.SHORT_DISTANCE_FLAT_RATE = 200
        self.FALLBACK_RATE_PER_MILE = 1.50
        self.FALLBACK_SURCHARGE_MULTIPLIER = 1.25
        self.TRANSCAR_WEIGHT = 0.7
        self.COMPETITOR_WEIGHT = 0.3

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

    def calculate_distance(self, origin_zip, dest_zip):
        """Calculate geodesic distance between two ZIP codes."""
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

    # def calculate_confidence_score(self, result):
    #     """Calculate confidence score based on data quality and availability."""
    #     score = 0

    #     # Data Volume (40% weight)
    #     transcar_count = result.get('transcar_data_count', 0)
    #     competitor_count = result.get('competitor_data_count', 0)
    #     total_loads = transcar_count + competitor_count

    #     if total_loads >= 6:
    #         score += 40
    #     elif total_loads == 5:
    #         score += 35
    #     elif total_loads == 4:
    #         score += 30
    #     elif total_loads == 3:
    #         score += 20
    #     elif total_loads == 2:
    #         score += 10
    #     elif total_loads == 1:
    #         score += 5
    #     else:
    #         score += 0

    #     # Data Recency (20% weight)
    #     if result['calculation_method'] == 'Historical Data':
    #         score += 20
    #     else:
    #         score += 2

    #     # Route Type Match (20% weight)
    #     origin_within = result.get('origin_within_metro', True)
    #     dest_within = result.get('dest_within_metro', True)

    #     if origin_within and dest_within:
    #         if result['calculation_method'] == 'Historical Data':
    #             score += 20
    #         else:
    #             score += 8
    #     elif origin_within or dest_within:
    #         score += 12
    #     else:
    #         score += 5

    #     # Data Source Quality (20% weight)
    #     if result['calculation_method'] == 'Historical Data':
    #         if transcar_count > 0 and competitor_count > 0:
    #             score += 9
    #         elif transcar_count > 0:
    #             score += 10
    #         elif competitor_count > 0:
    #             score += 8
    #     else:
    #         score += 2

    #     return min(100, score)

    def calculate_rate(self, origin_zip, dest_zip, vehicle_type="Car - Sedan"):
        """Main rate calculation method."""
        result = {
            'origin_zip': origin_zip,
            'dest_zip': dest_zip,
            'vehicle_type': vehicle_type
        }

        # Step 1: Calculate distance
        distance = self.calculate_distance(origin_zip, dest_zip)
        result['distance'] = distance

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
            result['confidence_score'] = 95
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

        # Step 5: Determine base rate
        if historical_rate is not None:
            result['base_rate'] = historical_rate * distance
            result['per_mile_rate'] = historical_rate
            result['calculation_method'] = 'Historical Data'
        else:
            fallback_rate = self.FALLBACK_RATE_PER_MILE
            if not origin_within and not dest_within:
                fallback_rate *= self.FALLBACK_SURCHARGE_MULTIPLIER
            result['base_rate'] = fallback_rate * distance
            result['per_mile_rate'] = fallback_rate
            result['calculation_method'] = 'Fallback Rate'

        # Step 6: Calculate surcharges
        origin_surcharge = 0 if origin_within else self.calculate_surcharge(origin_distance)
        dest_surcharge = 0 if dest_within else self.calculate_surcharge(dest_distance)

        result['origin_surcharge'] = origin_surcharge
        result['dest_surcharge'] = dest_surcharge
        result['total_surcharge'] = origin_surcharge + dest_surcharge

        # Step 7: Final rate calculation
        result['final_rate'] = result['base_rate'] + result['total_surcharge']

        # Step 8: Calculate confidence score
        # result['confidence_score'] = self.calculate_confidence_score(result)

        return result


@st.cache_data
def load_data():
    """Load and cache the POC data"""
    try:
        df_all = pd.read_excel("POC-Data-2.xlsx", sheet_name='Sheet1', header=None)

        # Extract sections
        sample_routes = df_all.iloc[2:6, 0:6].copy()
        sample_routes.columns = sample_routes.iloc[0]
        sample_routes = sample_routes.iloc[1:].dropna(how='all').reset_index(drop=True)

        metro_definitions = df_all.iloc[10:43, 0:7].copy()
        metro_definitions.columns = metro_definitions.iloc[0]
        metro_definitions = metro_definitions.iloc[1:].dropna(how='all').reset_index(drop=True)

        zipcode_assignment = df_all.iloc[44:54, 0:5].copy()
        zipcode_assignment.columns = zipcode_assignment.iloc[0]
        zipcode_assignment = zipcode_assignment.iloc[1:].dropna(how='all').reset_index(drop=True)

        transcar_data = df_all.iloc[55:75, 0:11].copy()
        transcar_data.columns = transcar_data.iloc[0]
        transcar_data = transcar_data.iloc[1:].dropna(how='all').reset_index(drop=True)

        competitor_data = df_all.iloc[76:, 0:11].copy()
        competitor_data.columns = competitor_data.iloc[0]
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
        st.error("POC-Data-2.xlsx file not found. Please ensure it's in the same directory as this app.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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

    # Initialize calculator
    calculator = RateCalculator(metro_definitions, zipcode_assignment, transcar_data, competitor_data)

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
    calculate_button = st.sidebar.button("🚀 Calculate Rate", type="primary", use_container_width=True)

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
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)

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
                    delta=f"{result['calculation_method']}"
                )

            # with col3:
            #     confidence_color = "normal"
            #     if result['confidence_score'] >= 70:
            #         confidence_color = "normal"
            #     elif result['confidence_score'] >= 40:
            #         confidence_color = "normal" 
            #     else:
            #         confidence_color = "inverse"

            #     st.metric(
            #         label="🎯 Confidence",
            #         value=f"{result['confidence_score']}%",
            #         delta="Data Quality"
            #     )

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
                st.plotly_chart(fig, use_container_width=True)

                # Breakdown table
                breakdown_df = pd.DataFrame([
                    ['Base Rate', f"${result['base_rate']:.2f}", f"${result['per_mile_rate']:.3f}/mile × {result['distance']:.1f} miles"],
                    ['Origin Surcharge', f"${result.get('origin_surcharge', 0):.2f}", 
                     f"Outside metro: {result.get('origin_distance_to_metro', 0):.1f} miles" if result.get('origin_surcharge', 0) > 0 else "Within metro area"],
                    ['Destination Surcharge', f"${result.get('dest_surcharge', 0):.2f}",
                     f"Outside metro: {result.get('dest_distance_to_metro', 0):.1f} miles" if result.get('dest_surcharge', 0) > 0 else "Within metro area"],
                    ['**Total Rate**', f"**${result['final_rate']:.2f}**", f"**Final transport rate**"]
                ], columns=['Component', 'Amount', 'Details'])

                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            with tab2:
                # Route visualization
                origin_coords = calculator.get_zipcode_coords(origin_zip)
                dest_coords = calculator.get_zipcode_coords(dest_zip)

                if None not in origin_coords and None not in dest_coords:
                    fig = create_route_visualization(origin_coords, dest_coords, origin_location, dest_location)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

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
                else:
                    st.warning("Unable to display map - coordinate data not available")

            with tab3:
                # Historical data analysis
                transcar_count = result.get('transcar_data_count', 0)
                competitor_count = result.get('competitor_data_count', 0)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("📊 Transcar Records", transcar_count, "70% weight")

                with col2:
                    st.metric("🏢 Competitor Records", competitor_count, "30% weight")

                with col3:
                    total_records = transcar_count + competitor_count
                    st.metric("📈 Total Records", total_records, "Available data")

                if total_records > 0:
                    st.success(f"✅ Historical data available! Using {total_records} records for calculation.")

                    if result.get('historical_per_mile_rate'):
                        st.info(f"📊 Calculated historical rate: **${result['historical_per_mile_rate']:.3f} per mile**")

                        # Data source breakdown
                        if transcar_count > 0 and competitor_count > 0:
                            weighted_rate = (0.7 * transcar_count + 0.3 * competitor_count) / total_records
                            st.write(f"Rate calculated using 70% Transcar + 30% Competitor weighting")
                        elif transcar_count > 0:
                            st.write(f"Rate based on {transcar_count} Transcar records (preferred source)")
                        else:
                            st.write(f"Rate based on {competitor_count} Competitor records")
                else:
                    st.warning(f"⚠️ No historical data available for this metro pair. Using fallback rate of ${calculator.FALLBACK_RATE_PER_MILE}/mile.")

            with tab4:
                # Detailed calculation information
                st.subheader("🔧 Algorithm Details")

                # Algorithm steps
                steps = [
                    ("1. Distance Calculation", f"{result['distance']:.2f} miles using geodesic coordinates"),
                    ("2. Short Distance Check", f"{'Applied $200 flat rate' if result['distance'] < 100 else 'Standard calculation applied'}"),
                    ("3. Metro Assignment", f"Origin: {result.get('origin_metro', 'Unknown')}, Destination: {result.get('dest_metro', 'Unknown')}"),
                    ("4. Historical Data Lookup", f"{transcar_count + competitor_count} records found"),
                    ("5. Base Rate Calculation", f"${result['base_rate']:.2f} using {result['calculation_method']}"),
                    ("6. Surcharge Application", f"${result['total_surcharge']:.2f} total surcharges"),
                    ("7. Final Rate", f"${result['final_rate']:.2f} (${result['per_mile_rate']:.3f}/mile)"),
                ]

                for step, description in steps:
                    st.write(f"**{step}**: {description}")

                # Configuration display
                st.subheader("⚙️ Algorithm Configuration")
                config_data = {
                    'Parameter': [
                        'Short Distance Threshold',
                        'Short Distance Flat Rate',
                        'Fallback Rate per Mile',
                        'Transcar Data Weight',
                        'Competitor Data Weight',
                        'Fallback Surcharge Multiplier'
                    ],
                    'Value': [
                        f"{calculator.SHORT_DISTANCE_THRESHOLD} miles",
                        f"${calculator.SHORT_DISTANCE_FLAT_RATE}",
                        f"${calculator.FALLBACK_RATE_PER_MILE}",
                        f"{calculator.TRANSCAR_WEIGHT*100}%",
                        f"{calculator.COMPETITOR_WEIGHT*100}%",
                        f"{calculator.FALLBACK_SURCHARGE_MULTIPLIER}x"
                    ]
                }
                st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Transport Rate Calculator! 👋

        This application calculates transport rates using a sophisticated algorithm that considers:

        - 📏 **Distance calculations** using geodesic coordinates
        - 🏙️ **Metro area assignments** with configurable radius
        - 📊 **Historical data integration** (70% Transcar + 30% Competitor)
        - 💰 **Smart surcharge system** for outside-metro locations

        ### 🚀 How to Use:
        1. Select your **origin location** from the dropdown
        2. Select your **destination location** from the dropdown  
        3. Choose your **vehicle type**
        4. Click **"Calculate Rate"** to get your quote

        ### 📍 Available Locations:
        """)

        # Display available locations
        for i, location in enumerate(locations, 1):
            st.write(f"{i}. {location}")

        # Sample routes display
        st.subheader("📋 Sample Routes from POC Data")
        display_routes = sample_routes.copy()
        display_routes['Route'] = display_routes.apply(
            lambda row: f"{row['ORIGIN CITY']}, {row['ORIGIN STATE']} → {row['DEST CITY']}, {row['DEST STATE']}", 
            axis=1
        )

        st.dataframe(
            display_routes[['Route', 'ORIGIN ZIPCODE', 'DEST ZIPCODE']].rename(columns={
                'ORIGIN ZIPCODE': 'Origin ZIP',
                'DEST ZIPCODE': 'Destination ZIP'
            }),
            width=True,
            hide_index=True
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "**Transport Rate Calculator POC** • "
        f"Loaded {len(metro_definitions)} metro areas • "
        f"{len(zipcode_assignment)} ZIP codes • "
        f"{len(transcar_data)} Transcar records • "
        f"{len(competitor_data)} competitor records"
    )


if __name__ == "__main__":
    main()
