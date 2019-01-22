import numpy as np

def map_per_label(label, prediction, pred_count):
    """Calculates mean average precison at n predictions for one label.

    By default, if pred_count is not specified, the function will calculate
    map for 5 predictions.

    Args:
        label: True label
        prediction: list of predictions for the label
        pred_count: number of predictions provided for the label

    Returns:
        Mean average precision at n predictions for one label.
    """
    try:
        return 1 / (prediction[:pred_count].index(label) + 1)
    except ValueError:
        return 0.0


def mean_average_precision(labels, predictions, pred_count=5):
    """Calculates mean average precison at n predictions per label.

    By default, if pred_count is not specified, the function will calculate
    map for min(number of predictions, 5).

    Args:
        labels: list of truth labels
        predictions: list of predictions for each label
        pred_count: number of predictions provided for each label

    Returns:
        Mean average precision at n predictions per label.
    """
    return np.mean([map_per_label(label=label,
                                  prediction=prediction,
                                  pred_count=pred_count)
                    for label, prediction in zip(labels, predictions)])
