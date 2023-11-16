# Import pandas library
import pandas as pd
import numpy as np
from pycaret.regression import *

# Read the csv file
# And convert text type 'Date' column into number type
df = pd.read_csv("./pre-processing/coin_price.csv", parse_dates=["date"])

# Remove 'tradecount' column and 'volume usdt' column
# df = df.drop(["tradecount", "volume usdt"], axis=1)
# target_Y = df["BTC_close"]
# data_X = df.drop(["id", "date", "BTC_close", "tradecount", "volume usdt"], axis=1)
data = df.drop(["id", "date", "tradecount", "volume usdt"], axis=1)

# 1행에서 32500행까지 삭제
data.drop(range(0, 32500), inplace=True)

# 인덱스 재설정
df.reset_index(drop=True, inplace=True)

# PyCaret regression
exp = setup(
    data,
    target="BTC_close",
)
best_model = compare_models()
tuned_model = tune_model(best_model)
# evaluate_model(tuned_model)
save_model(best_model, "my_first_pipeline")

# Write to the xlsx file
# df.to_excel("coin_price.xlsx", index=None, header=True)

#print(target_Y)
