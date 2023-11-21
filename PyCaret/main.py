# Import pandas library
import pandas as pd
import datetime
from pycaret.regression import *

def stamp(*args):
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(stamp + " :: ", *args)

# Read the csv file
stamp("Data loading")
# And convert text type 'Date' column into number type
df = pd.read_csv("./rawData/coin_price.csv", parse_dates=["date"])
stamp("Data loaded")

# Remove 'tradecount' column and 'volume usdt' column
stamp("Data processing")
# df = df.drop(["tradecount", "volume usdt"], axis=1)
# target_Y = df["BTC_close"]
# data_X = df.drop(["id", "date", "BTC_close", "tradecount", "volume usdt"], axis=1)
data = df.drop(["id", "date", "tradecount", "volume usdt"], axis=1)

# 1행에서 32500행까지 삭제
data.drop(range(0, 32500), inplace=True)

# 인덱스 재설정
df.reset_index(drop=True, inplace=True)
stamp("Data processed")

stamp("PyCaret regression")
# PyCaret regression
exp = setup(
    data,
    target="BTC_close",
    #use_gpu=True,
)

stamp("Generate models")
best_model = compare_models()

stamp("Model tuning")
tuned_model = tune_model(best_model)

#stamp("Evaluate models")
#evaluate_model(tuned_model)

stamp("Save best tuned model")
save_model(tuned_model, "2023_BTC_Price_1121_0945")

# Write to the xlsx file
# df.to_excel("coin_price.xlsx", index=None, header=True)

# stamp(target_Y)
