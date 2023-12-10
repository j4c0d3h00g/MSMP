import math
import numpy as np
from data_preprocessing import *
from functions import *
from performance_evaluation import *
from sklearn.utils import resample
import pandas as pd


if __name__ == "__main__":
    number_of_bootstraps = 5
    k = 4
    multiplication_factor = 0.5
    data, dataframe = get_data()
    most_frequent_keys, keyvalues_dataframe = find_most_common_keyvalues(data, 0)   # Possible to change the number of keys
    clean_dataframe = clean_data(dataframe, keyvalues_dataframe, most_frequent_keys)

    total_products_considered = 0

    result = pd.DataFrame(columns=['Bootstrap', 'F1-star', 'Fraction comparisons LSH', 'PQ', 'PC', 'F1', 'Fraction comparisons MSM', 'b', 'r', 'Threshold'])
    for bootstrap in range(number_of_bootstraps):
        bootstrap_dataframe = resample(clean_dataframe, replace=True, random_state=42)
        bootstrap_dataframe.drop_duplicates(inplace=True)
        bootstrap_dataframe.reset_index(inplace=True)
        # bootstrap_dataframe = bootstrap_dataframe.loc[0:199, :]

        percentage_products_considered = len(bootstrap_dataframe) / len(clean_dataframe)
        total_products_considered += len(bootstrap_dataframe)

        binary_vector_representations = create_binary_vector_representations(bootstrap_dataframe, True, most_frequent_keys)

        # binary_vector_representations_title = create_binary_vector_representations(bootstrap_dataframe, True, most_frequent_keys)
        # binary_vector_representations_keyvalues = create_binary_vector_representations(bootstrap_dataframe, False, most_frequent_keys)
        # binary_vector_representations = np.concatenate((binary_vector_representations_title, binary_vector_representations_keyvalues))

        signature_matrix = min_hashing(multiplication_factor, binary_vector_representations)
        bands, rows = possible_values_br(signature_matrix)

        # signature_matrix_title = min_hashing(multiplication_factor, binary_vector_representations_title)
        # bands_title, rows_title = possible_values_br(signature_matrix_title)
        #
        # binary_vector_representations_keyvalues = create_binary_vector_representations(bootstrap_dataframe, False, most_frequent_keys)
        # signature_matrix_keyvalues = min_hashing(1, binary_vector_representations_keyvalues)
        # non_inf_indices = []
        # n, m = signature_matrix_keyvalues.shape
        # for i in range(m):
        #     if not np.isinf(signature_matrix_keyvalues[:, i]).any(axis=0):
        #         non_inf_indices.append(i)
        #
        # signature_matrix_keyvalues = signature_matrix_keyvalues[:, ~np.isinf(signature_matrix_keyvalues).any(axis=0)]
        # bands_keyvalues, rows_keyvalues = possible_values_br(signature_matrix_keyvalues)

        for i in range(len(bands)):
            pairs_lsh = locality_sensitive_hashing(signature_matrix, bands[i], rows[i])
            pair_quality_lsh, pair_completeness_lsh, f1_star, fraction_of_comparisons_lsh = evaluate_performance(pairs_lsh, bootstrap_dataframe)
            number_of_duplicates = count_duplicates(bootstrap_dataframe)

            # pairs_keyvalues = find_pairs_keyvalues(bootstrap_dataframe, most_frequent_keys, 3)
            # pair_quality_keyvalues, pair_completeness_keyvalues, f1_keyvalues, fraction_of_comparisons_keyvalues = evaluate_performance(pairs_keyvalues, bootstrap_dataframe)
            # potential_pairs = pairs_lsh.union(pairs_keyvalues)
            # pair_quality_combined, pair_completeness_combined, f1_combined, fraction_of_comparisons_combined = evaluate_performance(potential_pairs, bootstrap_dataframe)

            dissimilarity_matrix = create_dissimilarity_matrix(bootstrap_dataframe, pairs_lsh, k, most_frequent_keys)
            threshold = optimal_threshold(dissimilarity_matrix, bootstrap_dataframe)
            pairs_clustering = clustering(dissimilarity_matrix, threshold)
            pair_quality_clustering, pair_completeness_clustering, f1, fraction_of_comparisons_clustering = evaluate_performance(pairs_clustering, bootstrap_dataframe)

            result.loc[bootstrap + i, ['Bootstrap']] = bootstrap + 1
            result.loc[bootstrap + i, ['F1-star']] = f1_star
            result.loc[bootstrap + i, ['Fraction comparisons LSH']] = fraction_of_comparisons_lsh
            result.loc[bootstrap + i, ['PQ']] = pair_quality_clustering
            result.loc[bootstrap + i, ['PC']] = pair_completeness_clustering
            result.loc[bootstrap + i, ['F1']] = f1
            result.loc[bootstrap + i, ['Fraction comparisons MSM']] = fraction_of_comparisons_clustering
            result.loc[bootstrap + i, ['b']] = bands[i]
            result.loc[bootstrap + i, ['r']] = rows[i]
            result.loc[bootstrap + i, ['Threshold']] = threshold

            print(f'Result of combination {i + 1} (out of {len(bands)} possible combinations) of bootstrap {bootstrap + 1} (of {number_of_bootstraps} total bootstraps):')
            print(result.iloc[bootstrap + i])

        # for i in range(len(bands_title)):
        #     for j in range(len(bands_keyvalues)):
        #         pairs_lsh_title = locality_sensitive_hashing(signature_matrix_title, bands_title[i], rows_title[i])
        #         pairs_lsh_keyvalues = locality_sensitive_hashing(signature_matrix_keyvalues, bands_keyvalues[j], rows_keyvalues[j])
        #         transformed_pairs_lsh_keyvalues = []
        #         for pair_keyvalues in pairs_lsh_keyvalues:
        #             product1_index = pair_keyvalues[0]
        #             product2_index = pair_keyvalues[1]
        #
        #             product1 = non_inf_indices[product1_index]
        #             product2 = non_inf_indices[product2_index]
        #             pair_tuple = tuple((product1, product2))
        #
        #             transformed_pairs_lsh_keyvalues.append(pair_tuple)
        #
        #         transformed_pairs_lsh_keyvalues = set(transformed_pairs_lsh_keyvalues)
        #
        #         pairs_lsh = pairs_lsh_title.union(transformed_pairs_lsh_keyvalues)
        #         pair_quality_lsh, pair_completeness_lsh, f1_star, fraction_of_comparisons_lsh = evaluate_performance(pairs_lsh, bootstrap_dataframe)
        #
        #         dissimilarity_matrix = create_dissimilarity_matrix(bootstrap_dataframe, pairs_lsh, 4)
        #         threshold = optimal_threshold(dissimilarity_matrix, bootstrap_dataframe)
        #         pairs_clustering = clustering(dissimilarity_matrix, threshold)
        #         pair_quality_clustering, pair_completeness_clustering, f1, fraction_of_comparisons_clustering = evaluate_performance(pairs_clustering, bootstrap_dataframe)
        #
        #         result.loc[bootstrap + i + j, ['Bootstrap']] = bootstrap + 1
        #         result.loc[bootstrap + i + j, ['F1-star']] = f1_star
        #         result.loc[bootstrap + i + j, ['Fraction comparisons LSH']] = fraction_of_comparisons_lsh
        #         result.loc[bootstrap + i + j, ['PQ']] = pair_quality_clustering
        #         result.loc[bootstrap + i + j, ['PC']] = pair_completeness_clustering
        #         result.loc[bootstrap + i + j, ['F1']] = f1
        #         result.loc[bootstrap + i + j, ['Fraction comparisons MSM']] = fraction_of_comparisons_clustering
        #         result.loc[bootstrap + i + j, ['b_title']] = bands_title[i]
        #         result.loc[bootstrap + i + j, ['r_title']] = rows_title[i]
        #         result.loc[bootstrap + i + j, ['b_keyvalues']] = bands_keyvalues[j]
        #         result.loc[bootstrap + i + j, ['r_keyvalues']] = rows_keyvalues[j]
        #         result.loc[bootstrap + i + j, ['Threshold']] = threshold
        #
        #         print(f'Result of combination {(i + 1) * (j + 1)} (out of {len(bands_title) * len(bands_keyvalues)} possible combinations) of bootstrap {bootstrap + 1} (of {number_of_bootstraps} total bootstraps):')
        #         print(result.iloc[bootstrap + i])

    average_percentage_products_considered = (total_products_considered / len(clean_dataframe)) / number_of_bootstraps
    print(average_percentage_products_considered)
    result.to_excel("result_bootstrapping.xlsx")

