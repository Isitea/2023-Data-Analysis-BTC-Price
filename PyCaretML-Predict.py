# Importing necessary libraries
import pandas as pd
import argparse

# Importing the custom AutoML module
from PyCaretLib.AutoML import apply_regression_model as apply_model

# Setting up command-line argument parser
arg_parser = argparse.ArgumentParser(
    description="AutoML script using PyCaret for HTC Analysis contest 2023"
)

# Adding command-line arguments for the model, input data file, and optional output file
arg_parser.add_argument(
    "model_path",
    type=str,
    help="Path to the trained model for prediction",
)
arg_parser.add_argument(
    "input_data",
    type=str,
    help="Path to the CSV file containing data for prediction",
)
arg_parser.add_argument(
    "output_file",
    type=str,
    nargs="?",
    help="Path to the CSV file to store predicted data",
)
args = arg_parser.parse_args()


# Main function to execute AutoML and generate predictions
def run_automl():
    # Reading input data from the specified CSV file
    input_data = pd.read_csv(args.input_data)

    # Applying the trained model to make predictions
    predictions = apply_model(input_data, args.model_path)

    # Renaming columns for clarity, assuming the prediction label is "BTC_close"
    result = predictions.rename(columns={"prediction_label": "BTC_close"})

    # Reordering columns for better organization
    result = result.reindex(
        columns=result.columns[0:2].tolist()
        + result.columns[-1:].tolist()
        + result.columns[2:-1].tolist()
    )

    # Saving the result to a CSV file, either the default "answer.csv" or the specified output file
    output_file = args.output_file if args.output_file else "regression.csv"
    result.to_csv(output_file, index=None, header=True)


# Calling the main function to start the AutoML process
run_automl()
