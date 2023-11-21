# Import pandas library
import pandas as pd
import datetime
import os
import sys

from PyCaretLib.AutoML import AutoML as ML


def recentHours(dataframe, hours=323):
    # Remove from row 1 to row 32500
    # Last data: 32823 (2023.09.23 16:00)
    # 323 hours = 13 days 11 hours => 2023.09.10 05:00 (32500)
    cropedData = dataframe.drop(range(0, 32823 - hours))

    # Reset index
    cropedData.reset_index(drop=True, inplace=True)

    return cropedData


def loadFile():
    # Read the csv file
    df = pd.read_csv("./rawData/coin_price.csv")

    # Remove 'id' column, 'data' column, 'tradecount' column and 'volume usdt' column
    data = df.drop(["id", "date", "tradecount", "volume usdt"], axis=1)

    return data


def trainer(data=loadFile(), count=1):
    for runs in range(count):
        timeStamp = str(datetime.datetime.now().strftime("%m%d_%H%M"))

        stdoutOrigin = sys.stdout
        sys.stdout = open("./" + timeStamp + ".log", "w", encoding="utf8")

        ML(
            cropedData=recentHours(dataframe=data, hours=323),
            model_name="2023_BTC_Price_" + timeStamp,
            use_gpu=False,
        )

        sys.stdout.close()
        sys.stdout = stdoutOrigin
        os.rename("./Feature Importance.png", "./" + timeStamp + ".png")

    return True


trainer(count=2)
