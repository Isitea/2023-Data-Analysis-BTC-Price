# Importing the pandas library and necessary functions from PyCaret for regression
from pycaret.regression import *


def generate_regression_model(
    cropped_data,
    model_name="default",
    use_gpu=False,
    polynomial_features=True,
    polynomial_degree=3,
):
    # Set up the PyCaret regression experiment
    experiment = setup(
        data=cropped_data,
        target="BTC_close",
        polynomial_features=polynomial_features,
        polynomial_degree=polynomial_degree,
        use_gpu=use_gpu,
    )

    # Compare regression models and select the best one
    best_model = compare_models()

    # Fine-tune the selected model
    tuned_model = tune_model(best_model)

    # Finalize the tuned model
    final_model = finalize_model(tuned_model)

    # Plot residual, error, learning, and feature importance plots
    plot_model(final_model, plot="residuals", save=True)
    plot_model(final_model, plot="error", save=True)
    plot_model(final_model, plot="learning", save=True)
    plot_model(final_model, plot="feature", save=True)

    # Save the final model with the specified name
    return save_model(final_model, model_name=model_name)


def apply_regression_model(
    input_data,
    model_name="default",
):
    # Load the pre-trained model
    model = load_model(model_name)

    # Make predictions using the loaded model
    predictions = predict_model(model, data=input_data)

    # Return the predictions
    return predictions
