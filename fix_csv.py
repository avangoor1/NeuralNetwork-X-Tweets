import csv

input_file = 'merged_values_keywords.csv'  # Path to your input CSV file
output_file = 'tweets_fixed.csv'  # Path to the output CSV file

with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = []

    for row in reader:
        # Quote the Tweet Text field if it contains a comma
        if ',' in row[3]:  # Assuming Tweet Text is the 4th column (index 3)
            row[3] = f'"{row[3]}"'
        rows.append(row)

with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)

print(f"CSV fixed and saved as {output_file}")
