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
    help="Number of recent data hours for training; set to 0 for entire range.",
)
parser.add_argument(
    "--GPU",
    action=argparse.BooleanOptionalAction,
    required=False,
    default=False,
    help="Enable GPU acceleration for training",
)
parser.add_argument(
    "--features",
    action=argparse.BooleanOptionalAction,
    required=False,
    default=False,
    help="Derive new features using existing numeric features",
)
parser.add_argument(
    "-D",
    "--degree",
    metavar="N",
    type=int,
    required=False,
    default=3,
    help="Degree of polynomial features; ignored when polynomial_features is False",
)

# Parse the command-line arguments
args = parser.parse_args()


# Function to filter recent hours of data from the dataframe
def filter_recent_hours(dataframe, hours=323):
    # Remove data older than specified hours
    if hours > 0:
        cropped_data = dataframe.drop(range(0, 32823 - hours))
    else:
        cropped_data = dataframe

    # Reset index
    cropped_data.reset_index(drop=True, inplace=True)

    return cropped_data


# Function to load the CSV file, remove unnecessary columns, and return the data
def load_file():
    # Read the csv file
    df = pd.read_csv("./rawData/coin_price.csv")

    # Remove unnecessary columns
    data = df.drop(["id", "date", "tradecount", "volume usdt"], axis=1)

    return data


# Function to train regression models using PyCaretLib.AutoML
def train_models(
    data=load_file(),
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
        if polynomial_features:
            if polynomial_degree < 1:
                polynomial_degree = 1
            model_name = model_name + "[PD" + str(polynomial_degree) + "]"

        # Print information about the current model generation
        print(
            "Generating model {current} / {total}".format(current=runs + 1, total=count)
        )
        if use_gpu:
            print("With GPU training")
        if polynomial_features:
            print(
                "With polynomial assumption: {degree}".format(degree=polynomial_degree)
            )

        # Redirect stdout to a log file for each model
        stdout_origin = sys.stdout
        sys.stdout = open("./" + model_name + ".log", "w", encoding="utf8")

        # Call the 'generateModel' function to train the model
        generateModel(
            cropped_data=filter_recent_hours(dataframe=data, hours=hours),
            model_name="2023_BTC_Price_" + model_name,
            use_gpu=use_gpu,
            polynomial_features=polynomial_features,
            polynomial_degree=polynomial_degree,
        )

        # Close the log file and reset stdout
        sys.stdout.close()
        sys.stdout = stdout_origin

        # Rename generated plots with the model-specific names
        os.rename(
            "./Feature Importance.png", "./" + model_name + " Feature Importance.png"
        )
        os.rename("./Prediction Error.png", "./" + model_name + " Prediction Error.png")
        os.rename("./Learning Curve.png", "./" + model_name + " Learning Curve.png")
        os.rename("./Residuals.png", "./" + model_name + " Residuals.png")

    return True


# Call the 'train_models' function with command-line arguments
train_models(
    count=args.repeat,
    hours=args.hours,
    polynomial_features=args.features,
    polynomial_degree=args.degree,
    use_gpu=args.GPU,
)
