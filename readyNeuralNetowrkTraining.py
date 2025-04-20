import csv
import re

def process_csv(input_file, output_file):
    results = []
    
    # Read the CSV file
    with open(input_file, mode='r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # Ensure the row has at least two fields
            if len(row) < 2:
                continue
            
            # Get the second-to-last field
            field_4 = row[-2]
            # Extract content inside triple quotes
            extracted_content = re.search(r'"""(.*?)"""', field_4, re.DOTALL)
            if extracted_content:
                field_4_cleaned = extracted_content.group(1)
            else:
                # If no triple quotes are found, fallback to the raw content
                field_4_cleaned = field_4
            # Ensure all triple quotes are removed (even in fallback content)
            field_4_cleaned = field_4_cleaned.replace('"""', '').strip()
            # Get the last field and clean it
            field_5 = row[-1].strip()
            # Add the cleaned fields to the results
            results.append([field_4_cleaned, field_5])
    
    # Write the results to a new CSV file
    with open(output_file, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(results)


# Example usage
input_csv_file = "trainingSet100.csv"  # Replace with your input CSV file name
output_csv_file = "output.csv"  # Replace with your desired output CSV file name

process_csv(input_csv_file, output_csv_file)
