# Import pandas library
import pandas as pd

# Read the csv file
# And convert text type 'Date' column into number type
df = pd.read_csv("coin_price.csv", parse_dates=['date'])

# Write to the xlsx file
df.to_excel("coin_price.xlsx", index=None, header=True)