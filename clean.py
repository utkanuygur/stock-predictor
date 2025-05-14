import pandas as pd

# Read your input file
df = pd.read_csv("sorted_by_ticker.csv")

# Desired column order
new_order = [
    'CIK', 'Ticker', 'Year', 'Quarter',
    'Total Revenue', 'Net Income', 'Operating Margin', 'EPS', 'Stock Price'
]

# Reorder columns
df = df[new_order]

# Write to output file
df.to_csv("cleaned_file2.csv", index=False)
