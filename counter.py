import pandas as pd

# Replace with the actual path to your CSV file
df = pd.read_csv("cleaned_file.csv")

# Count occurrences of each ticker
ticker_counts = df['Ticker'].value_counts()

# Print the results
print(ticker_counts)
