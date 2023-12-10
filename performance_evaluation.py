from functions import *
from data_preprocessing import get_data, clean_data, count_duplicates, possible_values_br


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


if __name__ == "__main__":
    data, dataframe = get_data()
    most_frequent_keys, keyvalues_dataframe = find_most_common_keyvalues(data, 0)
    dataframe = clean_data(dataframe, keyvalues_dataframe, most_frequent_keys)
    binary_vector_representations = create_binary_vector_representations(dataframe, True, most_frequent_keys)
    signature_matrix = min_hashing(0.5, binary_vector_representations)
    bands, rows = possible_values_br(signature_matrix)

    for i in range(len(bands)):
        pairs_lsh = locality_sensitive_hashing(signature_matrix, bands[i], rows[i])
        pair_quality_lsh, pair_completeness_lsh, f1_star, fraction_of_comparisons_lsh = evaluate_performance(pairs_lsh, dataframe)

        print('b:', bands[i], 'r:', rows[i])
        print('Pair quality:', pair_quality_lsh)
        print('Pair completeness:', pair_completeness_lsh)
        print('F1_star:', f1_star)
        print('Fraction of comparisons:', fraction_of_comparisons_lsh)

        dissimilarity_matrix = create_dissimilarity_matrix(dataframe, pairs_lsh, 4, most_frequent_keys)
        threshold = optimal_threshold(dissimilarity_matrix, dataframe)
        pairs_clustering = clustering(dissimilarity_matrix, threshold)
        pair_quality_clustering, pair_completeness_clustering, f1, fraction_of_comparisons_clustering = evaluate_performance(pairs_clustering, dataframe)

        print('b:', bands[i], 'r:', rows[i])
        print('Pair quality:', pair_quality_clustering)
        print('Pair completeness:', pair_completeness_clustering)
        print('F1:', f1)
        print('Fraction of comparisons:', fraction_of_comparisons_clustering)


