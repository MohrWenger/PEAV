import pandas as pd

"""
This file is used for validating our output in comparison with a mannualy segmented dataset
"""

TAGGER = "מתייגת"


def loss_1_0(output_vals,validated_vals, check_match_func):
    """
    This function gets two feature colums - one from our output and one from our test and calculates 0 - 1 loss
    according to a given function.
    :param output_vals: output of our algorithm feature (column)
    :param validated_vals: test set same feature (column)
    :param check_match_func: a function that recieves two values and compares their equality returns a bool
    :return: the sum of accurate answers
    """
    if len(output_vals) != len(validated_vals):
        return ("output len doesn't mach")
    correct_rate = 0
    for i in range(len(output_vals)):
        if check_match_func(output_vals[i], validated_vals[i]):
            correct_rate += 1
    return correct_rate

def is_p_sentence_in_t(output_val, test_val):
    pass

if __name__ == "__main__":
    validated_df = pd.read_csv("Test Set = PEAV - Sheet1.csv", error_bad_lines=False)
    our_output = pd.read_csv("output_learned.csv", error_bad_lines=False)




