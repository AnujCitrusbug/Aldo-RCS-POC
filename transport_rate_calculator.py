"""
Transport Rate Calculator Algorithm
===================================

A comprehensive rate calculation system for transport services that implements:
- Distance-based rate calculations
- Metro area assignments with ZIP code mapping
- Historical data integration (70% Transcar + 30% Competitor)
- Surcharge calculations for outside-metro locations
- Confidence score assessment
- Fallback rate logic

Author: Rate Calculator POC
Date: October 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')


class RateCalculator:
    """
    Main rate calculation engine that implements the transport rate algorithm.

    This class handles:
    - Loading and processing input data (metros, zipcodes, historical data)
    - Distance calculations using geodesic coordinates
    - Metro area assignments based on proximity and radius
    - Historical data lookup with weighted averages
    - Surcharge calculations for outside-metro locations
    - Confidence scoring based on data quality and availability
    """

    def __init__(self, metro_definitions, zipcode_assignment, transcar_data, competitor_data):
        """
        Initialize the rate calculator with required datasets.

        Args:
            metro_definitions (DataFrame): Metro areas with coordinates and radius
            zipcode_assignment (DataFrame): ZIP codes with coordinates
            transcar_data (DataFrame): Historical Transcar pricing data
            competitor_data (DataFrame): Historical competitor pricing data
        """
        self.metro_definitions = metro_definitions.copy()
        self.zipcode_assignment = zipcode_assignment.copy()
        self.transcar_data = transcar_data.copy()
        self.competitor_data = competitor_data.copy()

        # Convert numeric columns for calculations
        self._prepare_data()

        # Configuration constants
        self.SHORT_DISTANCE_THRESHOLD = 100  # miles
        self.SHORT_DISTANCE_FLAT_RATE = 200  # dollars
        self.FALLBACK_RATE_PER_MILE = 1.50   # dollars per mile
        self.FALLBACK_SURCHARGE_MULTIPLIER = 1.25  # for non-metro to non-metro
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
        """
        Get latitude and longitude coordinates for a given ZIP code.

        Args:
            zipcode: ZIP code to lookup

        Returns:
            tuple: (latitude, longitude) or (None, None) if not found
        """
        try:
            zipcode_int = int(float(str(zipcode)))
            zip_row = self.zipcode_assignment[self.zipcode_assignment['Zip_Code'] == zipcode_int]
            if not zip_row.empty:
                return float(zip_row.iloc[0]['Latitude']), float(zip_row.iloc[0]['Longitude'])
        except (ValueError, TypeError):
            pass
        return None, None

    def assign_metro(self, zipcode):
        """
        Assign a ZIP code to its nearest metro area.

        Args:
            zipcode: ZIP code to assign to metro

        Returns:
            tuple: (metro_name, is_within_radius, distance_to_metro)
        """
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
        """
        Calculate geodesic distance between two ZIP codes.

        Args:
            origin_zip: Origin ZIP code
            dest_zip: Destination ZIP code

        Returns:
            float: Distance in miles, 0 if coordinates not found
        """
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
        """
        Search for historical pricing data between metro pairs.

        Implements 70% Transcar + 30% Competitor weighting as specified.
        Searches bidirectionally for better data coverage.

        Args:
            origin_metro: Origin metro area name
            dest_metro: Destination metro area name

        Returns:
            tuple: (weighted_rate_per_mile, transcar_count, competitor_count)
        """
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
        """
        Calculate surcharge based on distance from nearest metro area.

        Surcharge schedule as per documentation:
        - 1-25 miles: $50
        - 26-50 miles: $75
        - 51-100 miles: $100
        - 101-150 miles: $150
        - 151-200 miles: $200
        - 200+ miles: $250

        Args:
            distance_to_metro: Distance in miles to nearest metro

        Returns:
            float: Surcharge amount in dollars
        """
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

    def calculate_confidence_score(self, result):
        """
        Calculate confidence score based on data quality and availability.

        Scoring factors (as per documentation):
        - Data Volume (40% weight): Number of historical records
        - Data Recency (20% weight): Age of historical data
        - Route Type Match (20% weight): Metro-to-metro vs mixed routes
        - Data Source Quality (20% weight): Transcar vs Competitor mix

        Args:
            result: Rate calculation result dictionary

        Returns:
            int: Confidence score (0-100%)
        """
        score = 0

        # Data Volume (40% weight)
        transcar_count = result.get('transcar_data_count', 0)
        competitor_count = result.get('competitor_data_count', 0)
        total_loads = transcar_count + competitor_count

        if total_loads >= 6:
            score += 40
        elif total_loads == 5:
            score += 35
        elif total_loads == 4:
            score += 30
        elif total_loads == 3:
            score += 20
        elif total_loads == 2:
            score += 10
        elif total_loads == 1:
            score += 5
        else:
            score += 0

        # Data Recency (20% weight) - Assuming recent data for POC
        if result['calculation_method'] == 'Historical Data':
            score += 20
        else:
            score += 2

        # Route Type Match (20% weight)
        origin_within = result.get('origin_within_metro', True)
        dest_within = result.get('dest_within_metro', True)

        if origin_within and dest_within:
            if result['calculation_method'] == 'Historical Data':
                score += 20
            else:
                score += 8
        elif origin_within or dest_within:
            score += 12
        else:
            score += 5

        # Data Source Quality (20% weight)
        if result['calculation_method'] == 'Historical Data':
            if transcar_count > 0 and competitor_count > 0:
                score += 9  # Mixed data source
            elif transcar_count > 0:
                score += 10  # Transcar preferred
            elif competitor_count > 0:
                score += 8   # Competitor only
        else:
            score += 2

        return min(100, score)

    def calculate_rate(self, origin_zip, dest_zip, vehicle_type="Car - Sedan"):
        """
        Main rate calculation method implementing the complete algorithm.

        Algorithm Flow:
        1. Calculate distance between origin and destination
        2. Check if short distance (< 100 miles) -> flat rate
        3. Assign metro areas to origin and destination
        4. Search for historical data between metro pairs
        5. Apply base rate (historical or fallback)
        6. Calculate and apply surcharges for outside-metro locations
        7. Calculate confidence score

        Args:
            origin_zip: Origin ZIP code
            dest_zip: Destination ZIP code
            vehicle_type: Vehicle type (default: "Car - Sedan")

        Returns:
            dict: Complete rate calculation results with breakdown
        """
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
            result['confidence_score'] = 95  # High confidence for flat rate
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
            # Fallback rate logic
            fallback_rate = self.FALLBACK_RATE_PER_MILE
            if not origin_within and not dest_within:
                fallback_rate *= self.FALLBACK_SURCHARGE_MULTIPLIER
            result['base_rate'] = fallback_rate * distance
            result['per_mile_rate'] = fallback_rate
            result['calculation_method'] = 'Fallback Rate'

        # Step 6: Calculate surcharges for outside-metro locations
        origin_surcharge = 0 if origin_within else self.calculate_surcharge(origin_distance)
        dest_surcharge = 0 if dest_within else self.calculate_surcharge(dest_distance)

        result['origin_surcharge'] = origin_surcharge
        result['dest_surcharge'] = dest_surcharge
        result['total_surcharge'] = origin_surcharge + dest_surcharge

        # Step 7: Final rate calculation
        result['final_rate'] = result['base_rate'] + result['total_surcharge']

        # Step 8: Calculate confidence score
        result['confidence_score'] = self.calculate_confidence_score(result)

        return result

    def calculate_multiple_routes(self, routes_data):
        """
        Calculate rates for multiple routes at once.

        Args:
            routes_data: List of dictionaries with 'origin_zip' and 'dest_zip' keys

        Returns:
            list: List of rate calculation results
        """
        results = []
        for route in routes_data:
            origin_zip = route.get('origin_zip')
            dest_zip = route.get('dest_zip')
            vehicle_type = route.get('vehicle_type', 'Car - Sedan')

            result = self.calculate_rate(origin_zip, dest_zip, vehicle_type)
            results.append(result)

        return results

    def export_results_to_csv(self, results, filename='rate_calculation_results.csv'):
        """
        Export calculation results to CSV file.

        Args:
            results: List of rate calculation results
            filename: Output CSV filename
        """
        export_data = []

        for result in results:
            row = {
                'Origin_ZIP': result['origin_zip'],
                'Destination_ZIP': result['dest_zip'],
                'Vehicle_Type': result['vehicle_type'],
                'Distance_Miles': round(result.get('distance', 0), 2),
                'Calculation_Method': result.get('calculation_method', ''),
                'Origin_Metro': result.get('origin_metro', ''),
                'Origin_Within_Metro': result.get('origin_within_metro', ''),
                'Origin_Distance_to_Metro': round(result.get('origin_distance_to_metro', 0), 2),
                'Destination_Metro': result.get('dest_metro', ''),
                'Destination_Within_Metro': result.get('dest_within_metro', ''),
                'Destination_Distance_to_Metro': round(result.get('dest_distance_to_metro', 0), 2),
                'Historical_Per_Mile_Rate': round(result.get('historical_per_mile_rate', 0) or 0, 3),
                'Transcar_Records_Used': result.get('transcar_data_count', 0),
                'Competitor_Records_Used': result.get('competitor_data_count', 0),
                'Base_Rate': round(result.get('base_rate', 0), 2),
                'Per_Mile_Rate': round(result.get('per_mile_rate', 0), 3),
                'Origin_Surcharge': round(result.get('origin_surcharge', 0), 2),
                'Destination_Surcharge': round(result.get('dest_surcharge', 0), 2),
                'Total_Surcharges': round(result.get('total_surcharge', 0), 2),
                'Final_Rate': round(result.get('final_rate', 0), 2),
                'Confidence_Score': result.get('confidence_score', 0)
            }
            export_data.append(row)

        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


def load_data_from_excel(filename, expand_zipcode_coverage=True):
    """
    Load and process data from the POC Excel file.

    Args:
        filename: Path to Excel file
        expand_zipcode_coverage: Whether to add missing ZIP codes for historical data

    Returns:
        tuple: (metro_definitions, zipcode_assignment, transcar_data, competitor_data)
    """
    # Load the Excel file
    df_all = pd.read_excel(filename, sheet_name='Sheet1', header=None)

    # Extract sections
    # Sample Routes (rows 2-5) - Not needed for calculator initialization but useful for testing
    sample_routes = df_all.iloc[2:6, 0:6].copy()
    sample_routes.columns = sample_routes.iloc[0]
    sample_routes = sample_routes.iloc[1:].dropna(how='all').reset_index(drop=True)

    # Metro Definitions (rows 10-42)
    metro_definitions = df_all.iloc[10:43, 0:7].copy()
    metro_definitions.columns = metro_definitions.iloc[0]
    metro_definitions = metro_definitions.iloc[1:].dropna(how='all').reset_index(drop=True)

    # Zipcode to Metro Assignment (rows 44-53)
    zipcode_assignment = df_all.iloc[44:54, 0:5].copy()
    zipcode_assignment.columns = zipcode_assignment.iloc[0]
    zipcode_assignment = zipcode_assignment.iloc[1:].dropna(how='all').reset_index(drop=True)

    # Transcar Historical Data (rows 55-74)
    transcar_data = df_all.iloc[55:75, 0:11].copy()
    transcar_data.columns = transcar_data.iloc[0]
    transcar_data = transcar_data.iloc[1:].dropna(how='all').reset_index(drop=True)

    # Competitor Data (rows 76+)
    competitor_data = df_all.iloc[76:, 0:11].copy()
    competitor_data.columns = competitor_data.iloc[0]
    competitor_data = competitor_data.iloc[1:].dropna(how='all').reset_index(drop=True)

    # Expand zipcode coverage for better historical data matching
    if expand_zipcode_coverage:
        additional_zipcodes = [
            # Houston area
            {'Zip_Code': 77013, 'City': 'Houston', 'State': 'TX', 'Latitude': 29.7604, 'Longitude': -95.3698},
            {'Zip_Code': 77044, 'City': 'Houston', 'State': 'TX', 'Latitude': 29.7604, 'Longitude': -95.3698},
            {'Zip_Code': 77084, 'City': 'Houston', 'State': 'TX', 'Latitude': 29.7604, 'Longitude': -95.3698},

            # Arlington area
            {'Zip_Code': 76001, 'City': 'Arlington', 'State': 'TX', 'Latitude': 32.7357, 'Longitude': -97.1081},

            # Austin area  
            {'Zip_Code': 73301, 'City': 'Austin', 'State': 'TX', 'Latitude': 30.2672, 'Longitude': -97.7431},

            # Corpus Christi area
            {'Zip_Code': 78401, 'City': 'Corpus Christi', 'State': 'TX', 'Latitude': 27.8006, 'Longitude': -97.3964},

            # Dallas area
            {'Zip_Code': 75201, 'City': 'Dallas', 'State': 'TX', 'Latitude': 32.7767, 'Longitude': -96.7970},
            {'Zip_Code': 75217, 'City': 'Dallas', 'State': 'TX', 'Latitude': 32.7767, 'Longitude': -96.7970},
            {'Zip_Code': 75236, 'City': 'Dallas', 'State': 'TX', 'Latitude': 32.7767, 'Longitude': -96.7970},

            # Other Texas cities
            {'Zip_Code': 79424, 'City': 'Lubbock', 'State': 'TX', 'Latitude': 33.5779, 'Longitude': -101.8552},
            {'Zip_Code': 76901, 'City': 'San Angelo', 'State': 'TX', 'Latitude': 31.4638, 'Longitude': -100.4370},
            {'Zip_Code': 76905, 'City': 'San Angelo', 'State': 'TX', 'Latitude': 31.4638, 'Longitude': -100.4370}
        ]

        zipcode_assignment = pd.concat([
            zipcode_assignment,
            pd.DataFrame(additional_zipcodes)
        ], ignore_index=True)

    return metro_definitions, zipcode_assignment, transcar_data, competitor_data, sample_routes


def main_example():
    """
    Example usage of the Rate Calculator.
    This demonstrates how to use the calculator with the POC data.
    """
    print("Transport Rate Calculator - POC Example")
    print("=" * 50)

    # Load data from Excel file
    try:
        filename = "POC-Data-2.xlsx"  # Update this path as needed
        metro_definitions, zipcode_assignment, transcar_data, competitor_data, sample_routes = load_data_from_excel(filename)

        print(f"Data loaded successfully!")
        print(f"Metro areas: {len(metro_definitions)}")
        print(f"ZIP codes: {len(zipcode_assignment)}")
        print(f"Transcar records: {len(transcar_data)}")
        print(f"Competitor records: {len(competitor_data)}")
        print()

    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        print("Please ensure the POC data file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Initialize the calculator
    calculator = RateCalculator(metro_definitions, zipcode_assignment, transcar_data, competitor_data)

    # Calculate rates for sample routes
    print("Calculating rates for sample routes:")
    print("-" * 40)

    for idx, route in sample_routes.iterrows():
        origin_zip = route['ORIGIN ZIPCODE']
        dest_zip = route['DEST ZIPCODE']

        result = calculator.calculate_rate(origin_zip, dest_zip)

        print(f"Route {idx + 1}: {route['ORIGIN CITY']}, {route['ORIGIN STATE']} → {route['DEST CITY']}, {route['DEST STATE']}")
        print(f"  ZIP: {origin_zip} → {dest_zip}")
        print(f"  Distance: {result['distance']:.2f} miles")
        print(f"  Method: {result['calculation_method']}")
        print(f"  Final Rate: ${result['final_rate']:.2f}")
        print()

    # Example: Calculate rate for custom route
    print("Example: Custom route calculation")
    print("-" * 40)
    custom_result = calculator.calculate_rate(75001, 77550)  # Addison to Galveston

    print(f"Custom route: {custom_result['origin_zip']} → {custom_result['dest_zip']}")
    print(f"Distance: {custom_result['distance']:.2f} miles")
    print(f"Final Rate: ${custom_result['final_rate']:.2f}")

    # Export all sample route results
    sample_route_data = []
    for idx, route in sample_routes.iterrows():
        sample_route_data.append({
            'origin_zip': route['ORIGIN ZIPCODE'],
            'dest_zip': route['DEST ZIPCODE'],
            'vehicle_type': 'Car - Sedan'
        })

    results = calculator.calculate_multiple_routes(sample_route_data)
    calculator.export_results_to_csv(results, 'sample_routes_results.csv')


if __name__ == "__main__":
    main_example()
