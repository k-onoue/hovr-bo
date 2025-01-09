import os
import pandas as pd
import ast

def process_file(filepath):
    """
    Process a single CSV file to remove brackets from numerical values.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)

        # Iterate over each column and clean up bracketed values
        for col in df.columns:
            if df[col].dtype == object:  # Only process object (string-like) columns
                df[col] = df[col].apply(lambda x: clean_brackets(x))

        # Save the cleaned file back
        df.to_csv(filepath, index=False)
        print(f"Processed and saved: {filepath}")

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def clean_brackets(value):
    """
    Remove brackets from a string representation of a list with one numeric element.
    """
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        try:
            # Convert string to a Python literal
            parsed = ast.literal_eval(value)
            # If it's a list with a single numeric value, return that value
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], (int, float)):
                return parsed[0]
        except (ValueError, SyntaxError):
            pass
    return value

def process_results_directory(directory):
    """
    Recursively process all CSV files in the given directory.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                process_file(filepath)

# Specify the directory to process
results_dir = "results"

# Process all CSV files in the directory
process_results_directory(results_dir)
