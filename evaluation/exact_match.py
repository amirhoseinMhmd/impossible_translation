def exact_match(prediction: list, actual: list):
    if len(prediction) != len(actual):
        raise ValueError("Prediction and actual lists must have the same length")

    if len(prediction) == 0:
        return 0.0
    matches = sum(pred == act for pred, act in zip(prediction, actual))
    accuracy = matches / len(prediction)

    return accuracy
