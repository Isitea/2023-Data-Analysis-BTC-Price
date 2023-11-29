# Importing necessary libraries
import pandas as pd
import argparse

# Importing the custom AutoML module
from PyCaretLib.AutoML import applyModel

# Setting up command-line argument parser
parser = argparse.ArgumentParser(
    description="AutoML script using PyCaret for HTC Analysis contest 2023"
)

# Adding command-line arguments for the model, input data file, and optional output file
parser.add_argument(
    "model",
    type=str,
    help="Trained model for prediction",
)
parser.add_argument(
    "question",
    type=str,
    help="The CSV file which holds data for prediction",
)
parser.add_argument(
    "answer",
    type=str,
    nargs="?",
    help="The CSV file which holds predicted data",
)
args = parser.parse_args()


# Main function to execute AutoML and generate predictions
def main():
    # Reading input data from the specified CSV file
    question = pd.read_csv(args.question)

    # Applying the trained model to make predictions
    answer = applyModel(question, args.model)

    # Renaming columns for clarity, assuming the prediction label is "BTC_close"
    result = answer.rename(columns={"prediction_label": "BTC_close"})

    # Reordering columns for better organization
    result = result.reindex(
        columns=result.columns[0:2].tolist()
        + result.columns[-1:].tolist()
        + result.columns[2:-1].tolist()
    )

    # Saving the result to a CSV file, either the default "answer.csv" or the specified output file
    if args.answer is None:
        result.to_csv("answer.csv", index=None, header=True)
    else:
        result.to_csv(args.answer, index=None, header=True)


# Calling the main function to start the AutoML process
main()
