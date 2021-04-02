import pandas as pd
import json
import numpy as np

"""
This file is used for validating our output in comparison with a mannualy segmented dataset
"""

TAGGER = "מתייגת"
OUR_CASENAME = "case_num"
TEST_CASENAME = "Case_Name"

PRED_MAIN_SENTENCE = "PENALTY_SENTENCE"
TEST_MAIN_SENTENCE = "עונש בפועל (המשפט הרלוונטי מהטקסט)"

PRED_TIME = "VOTED TIME"
TEST_TIME = "מספר חודשי עונש בפועל (מספר)"

NO_SENTENCE = " -- "

def loss_1_0(case_name_list, check_match_func, df, all_output, goal_output, relevant_pred_col, relevant_test_col):
    """
    This function gets two feature colums - one from our output and one from our test and calculates 0 - 1 loss
    according to a given function.
    :param output_vals: output of our algorithm feature (column)
    :param validated_vals: test set same feature (column)
    :param check_match_func: a function that recieves two values and compares their equality returns a bool
    :return: the sum of accurate answers
    """
    # if len(output_vals) != len(validated_vals):
    #     return ("output len doesn't mach")

    correct_rate = 0
    # for i in range(len(all_output[OUR_CASENAME])):
    #     all_output[OUR_CASENAME][i] =  all_output[OUR_CASENAME][i].strip("final_verdicts/")
    #     all_output[OUR_CASENAME][i] =  all_output[OUR_CASENAME][i]+"t"

    # all_output[OUR_CASENAME] = all_output[OUR_CASENAME].strip("final_verdicts")
    for i in range(len(relevant_cases)):
        predicted_line = all_output.loc[all_output[OUR_CASENAME] == relevant_cases[i]]
        goal_line = goal_output.loc[goal_output[TEST_CASENAME] == relevant_cases[i]]
        s = goal_line.shape
        print(s)
        if case_name_list[i] == "s00001581-359.txt":
            print("break_point")
        temp_df = check_match_func(predicted_line[OUR_CASENAME], goal_line[TEST_CASENAME],predicted_line, relevant_pred_col, goal_line,relevant_test_col, True)
        # temp_df = check_match_func(output_name[i], test_name[i],output_vals[i], validated_vals[i], True)
        if temp_df:
            df = pd.concat([df, temp_df[1]])
            if temp_df[0]:
                correct_rate += 1
    print(correct_rate)
    temp_df = df.loc[df["Error (difference)"] != "NO sentence"]
    # print(np.nanmean(np.array(temp_df["Error (difference)"])))
    # print(np.std(temp_df["Error (difference)"]))
    df.to_csv('validation result.csv', encoding= 'utf-8')
    return correct_rate


def time_comp(pred_case_name, goal_case_name, pred_line,relevent_pred_col, goal_line, relevent_test_col,write_to_df = False):
    pred_time = pred_line[relevent_pred_col].tolist()[0]
    goal_time = goal_line[relevent_test_col].tolist()[0]
    if pred_time != NO_SENTENCE:
        goal_time = float(goal_time)
        pred_time = float(pred_time)
        dist = np.abs(goal_time - pred_time)
    else:
        dist = "NO sentence"

    temp_df = pd.DataFrame([[pred_line[OUR_CASENAME].tolist()[0], pred_line[PRED_MAIN_SENTENCE].tolist()[0], pred_time,
                             goal_line[TEST_CASENAME].tolist()[0],goal_line[TEST_MAIN_SENTENCE].tolist()[0],goal_time,
                             dist, int(goal_time == pred_time)]],
                           columns=["predicted Casename", "predicted sentence", "predicted time",
                                    "Goal Casename", "Goal sentence", "Goal time",
                                    "Error (difference)","Error (Binary)"])
    if write_to_df:
        if goal_time == pred_time:
            return True, temp_df
        return False, temp_df


def is_t_sentence_in(output_name, test_name,output_val, test_val,write_to_df = False):
    if type(test_val.tolist()[0]) == str and len(output_val.tolist()) > 0:
        stripped_test = test_val.tolist()[0].strip(" \n,.")
        # print(output_name)

        stripped_output = output_val.tolist()[0].strip(" \n,.")
        output_name = output_name.tolist()[0]
        temp_df = pd.DataFrame([[output_name, stripped_output, test_name, stripped_test,stripped_test in stripped_output]],
                               columns=["predicted Casename","predicted after strip","Goal Casename","Goal after strip","Is t in P"])
        if write_to_df:
            if stripped_test in stripped_output:
                return True, temp_df
            return False, temp_df
        else:
            if stripped_test in stripped_output:
                return True
            return False
    else:
        if type(test_val) != str:
            print("didn't work for:",test_name," type test_val = ", type(test_val))
            print("text val = ", test_val.tolist()[0])


if __name__ == "__main__":
    df = pd.DataFrame()
    with open('test_case_filenames.txt') as json_file:
        relevant_cases = json.load(json_file)

    validated_df = pd.read_csv("Test Set - PEAV - Sheet1.csv", error_bad_lines=False)
    validated_df.sort_values(by=[TEST_CASENAME])
    # our_output = pd.read_csv("pipline on test set.csv", error_bad_lines=False)
    our_output = pd.read_csv("verdict_penalty.csv", error_bad_lines=False)
    our_output.sort_values(by=[OUR_CASENAME])
    # relevant_cases.sort()
    # loss_1_0(relevant_cases, is_t_sentence_in, df, our_output, validated_df, PRED_MAIN_SENTENCE, TEST_MAIN_SENTENCE)
    loss_1_0(relevant_cases, time_comp, df, our_output, validated_df, PRED_TIME, TEST_TIME)





