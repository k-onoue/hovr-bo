from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound


def get_acquisition_function(name):
    if name == "log_ei":
        return qLogExpectedImprovement
    elif name == "log_nei":
        return qLogNoisyExpectedImprovement
    elif name == "ucb":
        return qUpperConfidenceBound
    else:
        raise ValueError(f"Acquisition function {name} not recognized.")