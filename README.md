# Transport Rate Calculator - POC Implementation

A comprehensive Python implementation of the transport rate calculation algorithm that processes origin-destination routes and calculates accurate pricing based on historical data, metro area assignments, and configurable surcharges.

## Features

- **Distance-based Calculations**: Geodesic distance calculations between ZIP codes
- **Metro Area Assignment**: Automatic assignment of ZIP codes to nearest metro areas
- **Historical Data Integration**: 70% Transcar + 30% Competitor data weighting
- **Smart Fallback Logic**: Configurable fallback rates when no historical data is available  
- **Surcharge System**: Distance-based surcharges for outside-metro locations
- **Confidence Scoring**: Data quality assessment for each calculation
- **Batch Processing**: Calculate multiple routes simultaneously
- **CSV Export**: Export detailed results with complete breakdown

## Requirements

```bash
pip install pandas numpy geopy openpyxl
```

## Installation

1. Clone or download the repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your POC data file (`POC-Data-2.xlsx`) in the same directory
4. Run the script:
   ```bash
   python transport_rate_calculator.py
   ```
5. To run Streamlit app:
   ```bash
   streamlit run updated_streamlit_rate_calculator.py
   ```

## Data File Format

The system expects an Excel file with the following structure:

### Sheet Structure
- **SAMPLE ROUTES** (rows 2-5): Test routes for calculation
- **METRO DEFINITIONS** (rows 10-42): Metro areas with coordinates and radius
- **ZIPCODE TO METRO ASSIGNMENT** (rows 44-53): ZIP codes with coordinates
- **TRANSCAR HISTORICAL DATA** (rows 55-74): Historical Transcar pricing data
- **COMPETITOR DATA** (rows 76+): Historical competitor pricing data

### Required Columns

#### Metro Definitions
- State, Metro_Area, Population, Latitude, Longitude, radius_miles, is_port

#### Zipcode Assignment  
- Zip_Code, City, State, Latitude, Longitude

#### Transcar Historical Data
- Date, Pickup ZIP, Pickup City, Pickup State, Delivery ZIP, Delivery City, Delivery State, Vehicle Type, Distance, Total Carrier Price, Total Price per Mile

#### Competitor Data
- Origin City, Origin Zipcode, Origin State, Destination City, Dest Zipcode, Dest State, vehicle_type, Distance, competitor price minus profit, competitor price per Mile

## Usage

### Basic Usage

```python
from transport_rate_calculator import RateCalculator, load_data_from_excel

# Load data from Excel file
metro_definitions, zipcode_assignment, transcar_data, competitor_data, sample_routes = load_data_from_excel("POC-Data-2.xlsx")

# Initialize calculator
calculator = RateCalculator(metro_definitions, zipcode_assignment, transcar_data, competitor_data)

# Calculate rate for single route
result = calculator.calculate_rate(75001, 77550)  # Addison to Galveston
print(f"Final Rate: ${result['final_rate']:.2f}")
print(f"Confidence: {result['confidence_score']}%")
```

### Batch Processing

```python
# Define multiple routes
routes = [
    {'origin_zip': 75001, 'dest_zip': 77550, 'vehicle_type': 'Car - Sedan'},
    {'origin_zip': 76201, 'dest_zip': 77523, 'vehicle_type': 'SUV - Medium'},
    {'origin_zip': 77550, 'dest_zip': 79720, 'vehicle_type': 'Truck - Large'}
]

# Calculate all routes
results = calculator.calculate_multiple_routes(routes)

# Export to CSV
calculator.export_results_to_csv(results, 'my_results.csv')
```

### Custom Configuration

```python
# Initialize with custom parameters
calculator = RateCalculator(metro_definitions, zipcode_assignment, transcar_data, competitor_data)

# Modify configuration
calculator.SHORT_DISTANCE_THRESHOLD = 150  # Change short distance threshold
calculator.FALLBACK_RATE_PER_MILE = 1.75   # Adjust fallback rate
calculator.TRANSCAR_WEIGHT = 0.8           # Change data weighting
calculator.COMPETITOR_WEIGHT = 0.2
```

## Rate Calculation Algorithm

The system implements a comprehensive rate calculation algorithm:

### Step 1: Distance Calculation
- Uses geodesic distance between ZIP code coordinates
- Handles missing coordinates gracefully

### Step 2: Short Distance Check
- Routes under 100 miles (configurable) use flat rate of $200
- Bypasses complex calculations for short hauls

### Step 3: Metro Area Assignment
- Assigns ZIP codes to nearest metro areas
- Considers metro radius for within/outside determination
- Calculates exact distance to metro center

### Step 4: Historical Data Lookup
- Searches Transcar and Competitor data for matching metro pairs
- Applies 70% Transcar + 30% Competitor weighting
- Supports bidirectional route matching

### Step 5: Base Rate Calculation
- Uses historical data when available
- Falls back to configurable rate ($1.50/mile default)
- Applies 25% surcharge for non-metro to non-metro routes

### Step 6: Surcharge Application
Distance-based surcharges for outside-metro locations:
- 1-25 miles from metro: +$50
- 26-50 miles from metro: +$75
- 51-100 miles from metro: +$100
- 101-150 miles from metro: +$150
- 151-200 miles from metro: +$200
- 200+ miles from metro: +$250

### Step 7: Confidence Scoring
Multi-factor confidence assessment:
- **Data Volume (40%)**: Number of historical records
- **Data Recency (20%)**: Age of historical data  
- **Route Type Match (20%)**: Metro-to-metro vs mixed routes
- **Data Source Quality (20%)**: Transcar vs Competitor data mix

## Output Format

### Single Route Result
```python
{
    'origin_zip': 75001,
    'dest_zip': 77550,
    'vehicle_type': 'Car - Sedan',
    'distance': 279.53,
    'calculation_method': 'Historical Data',
    'origin_metro': 'Dallas, TX',
    'origin_within_metro': True,
    'origin_distance_to_metro': 12.87,
    'dest_metro': 'Galveston, TX', 
    'dest_within_metro': True,
    'dest_distance_to_metro': 1.39,
    'historical_per_mile_rate': 1.207,
    'transcar_data_count': 0,
    'competitor_data_count': 5,
    'base_rate': 337.52,
    'per_mile_rate': 1.207,
    'origin_surcharge': 0,
    'dest_surcharge': 0,
    'total_surcharge': 0,
    'final_rate': 337.52,
    'confidence_score': 48
}
```

### CSV Export Columns
- Origin_ZIP, Destination_ZIP, Vehicle_Type
- Distance_Miles, Calculation_Method
- Origin_Metro, Origin_Within_Metro, Origin_Distance_to_Metro
- Destination_Metro, Destination_Within_Metro, Destination_Distance_to_Metro
- Historical_Per_Mile_Rate, Transcar_Records_Used, Competitor_Records_Used
- Base_Rate, Per_Mile_Rate
- Origin_Surcharge, Destination_Surcharge, Total_Surcharges
- Final_Rate, Confidence_Score

## Configuration Parameters

### Algorithm Settings
```python
SHORT_DISTANCE_THRESHOLD = 100      # Miles threshold for flat rate
SHORT_DISTANCE_FLAT_RATE = 200      # Dollars for short distance routes
FALLBACK_RATE_PER_MILE = 1.50       # Default rate per mile
FALLBACK_SURCHARGE_MULTIPLIER = 1.25 # Extra multiplier for non-metro routes
TRANSCAR_WEIGHT = 0.7               # Weight for Transcar data
COMPETITOR_WEIGHT = 0.3             # Weight for Competitor data
```

### Surcharge Schedule
Configurable surcharge amounts based on distance to nearest metro area.

## Error Handling

The system handles various edge cases:
- Missing ZIP code coordinates
- Invalid input data
- Missing historical data
- Calculation errors
- File format issues

## Example Run Output

```
Transport Rate Calculator - POC Example
==================================================
Data loaded successfully!
Metro areas: 30
ZIP codes: 18
Transcar records: 17
Competitor records: 18

Calculating rates for sample routes:
----------------------------------------

Route 1: Addison, TX → Galveston, TX
  ZIP: 75001 → 77550
  Distance: 279.53 miles
  Method: Historical Data
  Final Rate: $337.52
  Confidence: 48%

Route 2: Denton, TX → Baytown, TX
  ZIP: 76201 → 77523
  Distance: 272.96 miles
  Method: Historical Data
  Final Rate: $407.04
  Confidence: 75%

Route 3: Galveston, TX → Big Spring, TX
  ZIP: 77550 → 79720
  Distance: 447.44 miles
  Method: Fallback Rate
  Final Rate: $746.16
  Confidence: 16%

Results exported to sample_routes_results.csv
```

## Troubleshooting

### Common Issues

1. **"Could not find POC-Data-2.xlsx"**
   - Ensure the Excel file is in the same directory as the Python script
   - Check file name spelling and extension

2. **"Unable to calculate distance - ZIP codes not found"**
   - Verify ZIP codes exist in the zipcode assignment data
   - Add missing ZIP codes with coordinates to the dataset

3. **Low confidence scores**
   - Indicates limited historical data for the route
   - Consider adding more historical data or adjusting confidence scoring

4. **Import errors**
   - Install required dependencies: `pip install pandas numpy geopy openpyxl`
   - Ensure Python version 3.6 or higher

### Performance Considerations

- Large datasets may require processing time for distance calculations
- Consider caching results for frequently calculated routes
- Batch processing is more efficient for multiple routes

## Extending the System

### Adding New Data Sources
1. Extend the `load_data_from_excel()` function to handle new data formats
2. Modify the historical data lookup methods
3. Update confidence scoring algorithms

### Custom Surcharge Rules
1. Override the `calculate_surcharge()` method
2. Implement custom business logic for specific scenarios
3. Add new configuration parameters

### Integration with APIs
1. Replace Excel data loading with API calls
2. Add real-time ZIP code coordinate lookup
3. Implement caching for performance optimization

## License

This is a Proof of Concept implementation. Please review and test thoroughly before production use.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Verify input data format matches requirements
3. Review algorithm parameters and configuration
4. Test with known good data to isolate issues

---

**Version:** 1.0  
**Last Updated:** October 2025  
**Python Version:** 3.6+
