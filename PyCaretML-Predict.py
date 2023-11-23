import pandas as pd
import argparse

from PyCaretLib.AutoML import applyModel

parser = argparse.ArgumentParser(
    description="AutoML script using PyCaret for HTC Analysis contest 2023"
)
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


def main():
    question = pd.read_csv(args.question)

    answer = applyModel(question, args.model)
    result = answer.rename(columns={"prediction_label": "BTC_close"})
    result = result.reindex(
        columns=result.columns[0:2].tolist()
        + result.columns[-1:].tolist()
        + result.columns[2:-1].tolist()
    )
    if args.answer is None:
        result.to_csv("answer.csv", index=None, header=True)
    else:
        result.to_csv(args.answer, index=None, header=True)

main()
