# Import necessary libraries and modules
import pandas as pd
import datetime
import os
import sys
import argparse

# Import the 'generateModel' function from the PyCaretLib.AutoML module
from PyCaretLib.AutoML import generateModel

# Set up command-line argument parser with descriptions for the AutoML script
parser = argparse.ArgumentParser(
    description="AutoML script using PyCaret for HTC Analysis contest 2023"
)

# Define command-line arguments for the script
parser.add_argument(
    "-R",
    "--repeat",
    metavar="N",
    type=int,
    required=False,
    default=1,
    help="Repeat count for generating regression model",
)
parser.add_argument(
    "-H",
    "--hours",
    metavar="N",
    type=int,
    required=False,
    default=323,
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
    default=3,
    help="Degree of polynomial features. For example, if an input sample is two dimensional and of the form [a, b], the polynomial features with degree = 2 are: [1, a, b, a^2, ab, b^2]. Ignored when polynomial_features is not True.",
)

# Parse the command-line arguments
args = parser.parse_args()


# Function to filter recent hours of data from the dataframe
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


# Function to load the CSV file, remove unnecessary columns, and return the data
def loadFile():
    # Read the csv file
    df = pd.read_csv("./rawData/coin_price.csv")

    # Remove 'id' column, 'data' column, 'tradecount' column and 'volume usdt' column
    data = df.drop(["id", "date", "tradecount", "volume usdt"], axis=1)

    return data


# Function to train regression models using PyCaretLib.AutoML
def trainer(
    data=loadFile(),
    count=1,
    hours=323,
    polynomial_features=False,
    polynomial_degree=3,
    use_gpu=False,
):
    # Loop to generate and train the specified number of models
    for runs in range(count):
        # Create a unique model name based on the current timestamp and parameters
        model_name = (
            str(datetime.datetime.now().strftime("%m%d_%H%M"))
            + " (H"
            + str(hours)
            + ")"
        )

        # Modify model name if polynomial features are used
        if polynomial_features == True:
            if polynomial_degree < 1:
                polynomial_degree = 1
            model_name = model_name + "[PD" + str(polynomial_degree) + "]"

        # Print information about the current model generation
        print(
            "Generating model(s) {current} / {total}".format(
                current=runs + 1, total=count
            )
        )
        if use_gpu == True:
            print("With GPU training")
        if polynomial_features == True:
            print(
                "With polynomial assumption: {degree}".format(degree=polynomial_degree)
            )

        # Redirect stdout to a log file for each model
        stdoutOrigin = sys.stdout
        sys.stdout = open("./" + model_name + ".log", "w", encoding="utf8")

        # Call the 'generateModel' function to train the model
        generateModel(
            cropedData=recentHours(dataframe=data, hours=hours),
            model_name="2023_BTC_Price_" + model_name,
            use_gpu=use_gpu,
            polynomial_features=polynomial_features,
            polynomial_degree=polynomial_degree,
        )

        # Close the log file and reset stdout
        sys.stdout.close()
        sys.stdout = stdoutOrigin

        # Rename generated plots with the model-specific names
        os.rename(
            "./Feature Importance.png", "./" + model_name + " Feature Importance.png"
        )
        os.rename("./Prediction Error.png", "./" + model_name + " Prediction Error.png")
        os.rename("./Learning Curve.png", "./" + model_name + " Learning Curve.png")
        os.rename("./Residuals.png", "./" + model_name + " Residuals.png")

    return True


# Call the 'trainer' function with command-line arguments
trainer(
    count=args.repeat,
    hours=args.hours,
    polynomial_features=args.features,
    polynomial_degree=args.degree,
    use_gpu=args.GPU,
)
