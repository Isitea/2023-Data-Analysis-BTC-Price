# Import pandas library
import pandas as pd
import datetime
import os
import sys
import argparse

from PyCaretLib.AutoML import AutoML as ML

parser = argparse.ArgumentParser(
    description="AutoML script using PyCaret for HTC Analysis contest 2023"
)
parser.add_argument(
    "-R",
    "--repeat",
    metavar="N",
    type=int,
    required=False,
    help="Repeat count for generating regression model",
)
parser.add_argument(
    "-H",
    "--hours",
    metavar="N",
    type=int,
    required=False,
    help="The number of recent data for training can vary, including the option of using the entire range when set to 0.",
)
parser.add_argument(
    "--GPU",
    action=argparse.BooleanOptionalAction,
    required=False,
    default=False,
    help="When set to True, use GPU acceleration for training",
)
parser.add_argument(
    "--features",
    action=argparse.BooleanOptionalAction,
    required=False,
    default=False,
    help="When set to True, new features are derived using existing numeric features.",
)
parser.add_argument(
    "-D",
    "--degree",
    metavar="N",
    type=int,
    required=False,
    help="Degree of polynomial features. For example, if an input sample is two dimensional and of the form [a, b], the polynomial features with degree = 2 are: [1, a, b, a^2, ab, b^2]. Ignored when polynomial_features is not True.",
)
args = parser.parse_args()


def recentHours(dataframe, hours=323):
    # Remove from row 1 to row 32500
    # Last data: 32823 (2023.09.23 16:00)
    # 323 hours = 13 days 11 hours => 2023.09.10 05:00 (32500)
    if hours > 0:
        cropedData = dataframe.drop(range(0, 32823 - hours))
    else:
        cropedData = dataframe

    # Reset index
    cropedData.reset_index(drop=True, inplace=True)

    return cropedData


def loadFile():
    # Read the csv file
    df = pd.read_csv("./rawData/coin_price.csv")

    # Remove 'id' column, 'data' column, 'tradecount' column and 'volume usdt' column
    data = df.drop(["id", "date", "tradecount", "volume usdt"], axis=1)

    return data


def trainer(
    data=loadFile(),
    count=1,
    hours=323,
    polynomial_features=False,
    polynomial_degree=3,
    use_gpu=False,
):
    for runs in range(count):
        model_name = (
            str(datetime.datetime.now().strftime("%m%d_%H%M"))
            + " (H"
            + str(hours)
            + ")"
        )

        if polynomial_features == True:
            if polynomial_degree < 1:
                polynomial_degree = 1
            model_name = model_name + "[PD" + str(polynomial_degree) + "]"

        print(
            "Generating model(s) {current} / {total}".format(
                current=runs + 1, totral=count
            )
        )
        if use_gpu == True:
            print("With GPU training")
        if polynomial_features == True:
            print(
                "With polynomial assumption: {degree}".format(degree=polynomial_degree)
            )

        stdoutOrigin = sys.stdout
        sys.stdout = open("./" + model_name + ".log", "w", encoding="utf8")

        ML(
            cropedData=recentHours(dataframe=data, hours=hours),
            model_name="2023_BTC_Price_" + model_name,
            use_gpu=use_gpu,
            polynomial_features=polynomial_features,
            polynomial_degree=polynomial_degree,
        )

        sys.stdout.close()
        sys.stdout = stdoutOrigin
        os.rename(
            "./Feature Importance.png", "./" + model_name + " Feature Importance.png"
        )
        os.rename("./Prediction Error.png", "./" + model_name + " Prediction Error.png")
        os.rename("./Learning Curve.png", "./" + model_name + " Learning Curve.png")
        os.rename("./Residuals.png", "./" + model_name + " Residuals.png")

    return True


trainer(
    count=args.repeat,
    hours=args.hours,
    polynomial_features=args.features,
    polynomial_degree=args.degree,
    use_gpu=args.GPU,
)
