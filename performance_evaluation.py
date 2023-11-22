from functions import create_binary_vector_representations, min_hashing, locality_sensitive_hashing, create_dissimilarity_matrix, clustering
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


if __name__ == "__main__":
    _, dataframe = get_data()
    dataframe = clean_data(dataframe)
    binary_vector_representations = create_binary_vector_representations(dataframe)
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

        dissimilarity_matrix = create_dissimilarity_matrix(dataframe, pairs_lsh, 4)
        pairs_clustering = clustering(dissimilarity_matrix, 0.05)
        pair_quality_clustering, pair_completeness_clustering, f1, fraction_of_comparisons_clustering = evaluate_performance(pairs_clustering, dataframe)

        print('b:', bands[i], 'r:', rows[i])
        print('Pair quality:', pair_quality_clustering)
        print('Pair completeness:', pair_completeness_clustering)
        print('F1:', f1)
        print('Fraction of comparisons:', fraction_of_comparisons_clustering)


