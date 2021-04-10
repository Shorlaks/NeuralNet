import math


# Regression Loss Functions
def squared_error(label, output):
    """
    *Preferred loss function if the distribution of the target variables is gaussian.
    Large errors are penalized more then small ones.
    :param label: List: target variables.
    :param output: List: variables from output layer.
    :return: Float: error.
    """
    squared_sum = 0
    for l, o in zip(label, output):
        squared_sum += math.pow(l - o, 2)
    error = squared_sum
    return error


def absolute_error(label, output):
    """
    *Preferred loss function if the distribution of the target variables is multi_modal.
    Outliers don't play a big role.
    :param label: List: target variables.
    :param output: List: variables from output layer
    :return: Float: error
    """
    abs_sum = 0
    for l, o in zip(label, output):
        abs_sum += abs(l - o)
    error = abs_sum
    return error


def squared_logarithmic_error(label, output):
    """
    *Preferred loss function if the distribution of the target variables is gaussian.
    Large errors and small errors are penalized almost the same.
    The loss can be interpreted as a measure of the ratio between the true and predicted values.
    :param label: List: target variables.
    :param output: List: variables from output layer
    :return: Float: error
    """
    squared_logarithmic_sum = 0
    for l, o in zip(label, output):
        squared_logarithmic_sum += math.pow(math.log10(l + 1) - math.log10(o + 1), 2)
    error = squared_logarithmic_sum
    return error


# Binary Classification Loss Functions
def binary_cross_entropy(label, output):
    """
    :param label: List: target variable (a list with only 1 item).
    :param output: List: variable from output layer (a list with only 1 item).
    :return: Float: error
    """
    error = (label[0] * math.log10(output[0])) + ((1 - label[0]) * math.log10(1 - output[0]))
    return -1 * error


# Multi-Class Classification Loss Functions
def multi_class_cross_entropy(label, output):
    """
    :param label: List: target variables.
    :param output: List: variables from output layer
    :return: Float: error
    """
    logarithmic_sum = 0
    for l, o in zip(label, output):
        logarithmic_sum += (l * math.log10(o)) + ((1 - l) * math.log10(1 - o))
    error = logarithmic_sum / len(label)
    return -1 * error


# Derivatives
def squared_error_derivative(label, output):
    return -1 * (label - output)


def binary_cross_entropy_derivative(label, output):
    output += 0.0000000000001
    return (output-label) / (output * (1 - output))


def absolute_error_derivative(label, output):
    return 1 if label > output else 0
