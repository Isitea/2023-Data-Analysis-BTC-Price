# Import pandas library
from pycaret.regression import *


def AutoML(
    cropedData,
    model_name="default",
    use_gpu=False,
    polynomial_features=True,
    polynomial_degree=3,
):
    # PyCaret regression
    exp = setup(
        cropedData,
        target="BTC_close",
        polynomial_features=polynomial_features,
        polynomial_degree=polynomial_degree,
        use_gpu=use_gpu,
    )

    best_model = compare_models()
    tuned_model = tune_model(best_model)
    final_model = finalize_model(tuned_model)
    plot_model(final_model, plot="residuals", save=True)
    plot_model(final_model, plot="error", save=True)
    plot_model(final_model, plot="learning", save=True)
    plot_model(final_model, plot="feature", save=True)

    return save_model(final_model, model_name=model_name)
