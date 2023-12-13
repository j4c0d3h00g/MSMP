from functions import *


def evaluate_performance(potential_pairs, dataframe):
    duplicates_found = 0

    for pair in potential_pairs:
        product1 = pair[0]
        product2 = pair[1]

        if dataframe['modelID'][product1] == dataframe['modelID'][product2]:
            duplicates_found += 1

    number_of_duplicates = count_duplicates(dataframe)
    number_of_comparisons = len(potential_pairs) + 0.000001
    pair_quality = duplicates_found / number_of_comparisons
    pair_completeness = duplicates_found / number_of_duplicates

    f1 = (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness + 0.000001)

    n = len(dataframe)
    number_of_possible_comparisons = (n * (n - 1)) / 2
    fraction_of_comparisons = number_of_comparisons / number_of_possible_comparisons

    return pair_quality, pair_completeness, f1, fraction_of_comparisons


def optimal_threshold(dissimilarity_matrix, dataframe):
    optimal_f1 = 0
    optimal_threshold = 0

    for i in range(21):
        threshold = i / 20
        pairs = clustering(dissimilarity_matrix, threshold)
        _, _, f1, _ = evaluate_performance(pairs, dataframe)

        if f1 > optimal_f1:
            optimal_threshold = threshold
            optimal_f1 = f1

    return optimal_threshold
