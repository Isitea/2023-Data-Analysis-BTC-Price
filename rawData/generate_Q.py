# Import pandas library
import pandas as pd

# Read the csv file
# And convert text type 'Date' column into number type
df = pd.read_csv("coin_price.csv", parse_dates=["date"])
# df.loc[:, ["BTC_close", "tradecount", "volume usdt"]] = ""
df.drop(["BTC_close"], axis=1, inplace=True)

df.drop(range(0, 32823 - 5), inplace=True)

# Write to the xlsx file
df.to_csv("Q.csv", index=None, header=True)
