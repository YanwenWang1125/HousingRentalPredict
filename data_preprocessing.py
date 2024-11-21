import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import re
from datetime import datetime


# ================= Data Cleaning =================

def remove_useless_data(data):
    # Remove the address number using regular expression
    data["address"] = list(map(lambda x: re.sub(r'^\d+\s+', '', str(x)), data.address))
    # remove link column
    data = data.drop(columns=["link"])
    return data

# Convert values: replace "none" and NaN with 0, convert others to float        
def baths_cleaning(data, column_name):
    data[column_name] = data[column_name].map(
        lambda x: float(x) if x != "none" and not pd.isna(x) else 0
    )
    return data

## Cleans and converts the beds column to integers.
def beds_cleaning(data, column_name):
    """
    This function cleans and converts the 'beds' column to integers.
    It handles various formats including:
    - '2 Beds', '3 Beds', etc.: Converted to their respective integer values
    - 'Studio': Converted to 0
    - 'none Beds': Converted to 0
    - NaN values: Converted to 0
    
    The function uses a helper function 'get_num_of_bed' to extract the numeric value
    from strings like '2 Beds' or '1 Bed'.
    """
    # Covert string value to integer
    def get_num_of_bad(value):
        value = value.replace("Beds", "").replace("Bed", "").strip()
        return int(value)
    
    data[column_name] = data[column_name].map(
        # Replaces 'none Beds' and 'Studio' with 0.
        lambda x: get_num_of_bad(x) if x != "none Beds"and x != "Studio" and not pd.isna(x) else 0 
    )
    return data

def clean_sq_feet(value):
    """
    Cleans and converts the 'sq_feet' column to numeric values.

    This function handles various formats and special cases:
    1. Removes parentheses and their contents.
    2. Handles addition ('+') and multiplication ('*') operations:
       - For large numbers with addition (e.g., 1500900+600), it takes the first 4 or 5 digits.
       - Evaluates expressions like '2,150 + 950' or '10 * 12'.
    3. Handles ranges with hyphens ('-'):
       - Calculates the average of the range (e.g., '500-600' becomes 550).
       - For multiple hyphens, uses only the first and last number.
    4. Handles comma-separated values:
       - For multiple comma-separated values, sums them up.
       - For single comma as thousand separator, removes it.
    5. Removes non-numeric characters except '+', '-', '*', '.', and '×'.
    6. Replaces '×' with '*' for multiplication.
    7. Strips leading/trailing '+', '-', '*', and '.' characters.

    Special cases:
    - Handles professionally developed basements (e.g., '2,150 + 950 SF').
    - Manages multiple area descriptions (e.g., '1300 sft Basement, 350 sft Master bedroom').

    Returns:
    float: The cleaned and converted square footage value.
    """
    # Function implementation follows...

    # Remove parentheses and their contents
    value = re.sub(r'\([^)]*\)', '', value)

    # Remove any non-numeric characters except '+' and '-' and "*" and "." and "×"
    value = re.sub(r'[^\d+\-*.,×]', '', value)

    # replace specical mulitiplation sign
    value = value.replace("×", '*')
    
    # Handle cases with '+' or '-' at the beginning or end
    value = value.strip('+-*.')
    
    # If there's a '+', '*' evaluate the expression
    if '+' in value or '*' in value:
        # Handle large numbers with addition
        # eg, 1500900+600 happened in index series[13651], it means 1500 = 900 + 600
        if '+' in value:
            # remove , in the addition case,
            # eg, 2,150 + 950 SF (professionally developed basement)
            value = value.replace(",", "")
            parts = value.split('+')
            if len(parts[0]) > 4:  # If the first part is a large number (more than 4 digits)
                if len(parts[1]) == 4 and len(parts[0]) == 9:
                    return float(parts[0][:5])  # Return the first 5 digits of the large number
                elif len(parts[1]) == 5:
                    return float(parts[0][:5])  # Return the first 5 digits of the large number
                else:
                    return float(parts[0][:4])  # Return the first 4 digits of the large number
        # rest are '+' and '*'
        return float(eval(value))
    
    # if there's a '-', find the avergae
    elif '-' in value:
        # If there's more than one '-', keep only the first and last number
        if value.count('-') > 1:
            numbers = value.split('-')
            value = f"{numbers[0]}-{numbers[-1]}"
        return sum(map(float, value.split('-'))) / 2
    
    elif ',' in value:
        # Handle cases with commas
        if value.count(',') > 1:
            # Example: "1300 sft Basement, 350 sft Master bedroom, 800 sft Living and Dinning Room"
            # Replace commas with plus signs and strip any leading/trailing plus signs or commas
            value = value.replace(",", "+").strip("+,")
            # Return the total area as a float
            return float(eval(value))
        else:
            # 800 sqft for Condo, 150 sqft for balcony
            # 800, 150
            if len(value.split(',')[0]) > 2:
                value = value.replace(",", "+").strip("+,")
                return float(eval(value))
            
            # If there's only one comma, it's likely a thousands separator
            # Remove the comma and convert to float
            return float(value.replace(",", ''))
    
    # eg, 1200 s.f. up & 1000 s.f. developed lower level, 1200..1000
    elif value.count('.') > 1:
        # Convert all '.' to '+', and if '..' are together, make it one '+'
        value = value.replace('.', '+')
        # Remove any consecutive '+' signs
        # "1++2" would become "1+2"
        value = re.sub(r'\++', '+', value)
        # Strip any leading/trailing '+' signs
        value = value.strip('+')
        # Evaluate the expression
        return float(eval(value))
    
    # 1100 up and 1100 down
    # 11001100
    if len(value) > 4:
        return float(value[:4])

    # If it's a simple number, convert to int
    # if the value is nan, return np.nan and handle it later on
    return float(value) if value != "" else np.nan

# ================= Convert categorical to Numerical =================

def convert_availability_date(date_str):
    '''
    Convert categorical columns to numeric:
        Assign 1 for 'immediate'
        Assign 0 for 'no vacancy'
        Assign NaN for 'call for availability'
        Calculate the difference between the available date and the current date for other values
        Return NaN for missing values
        
    The missing value will handle in the later session
    '''
    if pd.isna(date_str):
        return np.nan
    
    date_str = str(date_str).lower().strip()
    if date_str == 'immediate':
        return 1
    elif date_str == 'no vacancy':
        return 0
    elif date_str == 'call for availability':
        return np.nan
    
    try:
        # Try to parse the date
        date = datetime.strptime(date_str, '%B %d')
        target_date = datetime(2024, 6, 1)  # June 1, 2024
        date = date.replace(year=datetime.now().year)
        
        # Calculate the number of days between the date and July 1
        days_difference = (date - target_date).days
        if days_difference < 1:
            return 1
        else:
            return days_difference
    except ValueError:
        # If parsing fails, return NaN
        return np.nan

def convert_column_to_numerical(data, column_name, convert_func, level_list):
    # Update the column in the original data with the converted values
    # Convert to str first in order to identify nan value
    # Lambda Function: Used in the apply method to pass level_list to smoke_convert.
    # convert_func(x, level_list): Calls smoke_convert with both the value and level_list.
    data[column_name] = data[column_name].astype(str).apply(lambda x: convert_func(x, level_list))
    return data

def smoke_convert(value, level_list):
    # Convert values to integers based on their index in the list
    return int(level_list.index(value))

def pet_convert(value, level_list):
    # Convert values to integers based on their index in the list
    return int(level_list.index(value))

def furnishing_convert(value, level_list):
    # Convert values to integers based on their index in the list
    return int(level_list.index(value))

def categorical_encoder(data, categorical_list):
    ordinal_encoder = OrdinalEncoder()
    data[categorical_list] = ordinal_encoder.fit_transform(data[categorical_list])
    return data

# ================= Handle missing var =================

def imputing_missing_data(data, categorical_list):
    
    imputer = SimpleImputer()

    # Apply the imputer to the specified columns and update the DataFrame
    data[categorical_list] = imputer.fit_transform(data[categorical_list])

    # # Calculate the number of missing values for each column after imputation
    # list_of_missing_value = [data[col].isnull().sum() for col in categorical_list]

    # # Create a dictionary to store the count of missing values for each column
    # dict_of_missing_value = {col: missing_num for col, missing_num in zip(categorical_list, list_of_missing_value)}

    # # Print the number of missing values for each column after imputation
    # print(f"Missing value for each column after Imputation:\n{dict_of_missing_value}")
    return data


# ===================== Main Function =========================
def process_data(data):

    # ================= Data Cleaning =================
    # remove the address number and link column
    data = remove_useless_data(data)
    
    # Clean bath room column
    data = baths_cleaning(data, 'baths')
    
    # Clean bed room column
    data = beds_cleaning(data, 'beds')
    
    # Clean sq feet column
    sq_feet = data.sq_feet.copy()
    sq_feet = sq_feet.astype(str)
    sq_feet = sq_feet.apply(clean_sq_feet)
    data['sq_feet'] = sq_feet
    

    # ================= Convert categorical to Numerical =================
    #Convert the "availability_date" category to a numerical format
    data['availability_date'] = data['availability_date'].apply(convert_availability_date)
    
    # Convert the "smoking" category to a numerical format
    smoking_level_list = ['Non-Smoking', 'Negotiable', "nan", "Smoke Free Building", 'Smoking Allowed']
    data = convert_column_to_numerical(data, 'smoking', smoke_convert, smoking_level_list)
    
    # Convert the "dog" category to a numerical format
    pet_level_list = ['False', "nan", 'True']
    data = convert_column_to_numerical(data, 'dogs', pet_convert, pet_level_list)
    
    # Convert the "cat" category to a numerical format
    data = convert_column_to_numerical(data, 'cats', pet_convert, pet_level_list)

    # Convert the "furnishing" category to a numerical format
    furnishing_level_list = ['Unfurnished', "Unfurnished, Negotiable", 'Negotiable', 'Furnished']
    data = convert_column_to_numerical(data, 'furnishing', furnishing_convert, furnishing_level_list)
    
    # Convert location from categorical to numerical
    location_category = ['city', 'province', 'address']
    data = categorical_encoder(data, location_category)

    # Convert lease term and type from categorical to numerical
    lease_term_and_type = ['lease_term', 'type']
    data = categorical_encoder(data, lease_term_and_type)


    # ================= Handle missing var =================
    column_to_fill = ['lease_term', 'sq_feet', 'availability_date']
    data = imputing_missing_data(data, column_to_fill)

    return data


# TEST
original_data = pd.read_csv('rentfaster.csv')
result = process_data(original_data)
print(result)