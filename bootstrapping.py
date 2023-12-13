from performance_evaluation import *
from sklearn.utils import resample
import pandas as pd


if __name__ == "__main__":
    number_of_bootstraps = 5
    number_of_keys = 0              # Number of most occurring keys for which the information is added to the product description (z).
    k = 4                           # Shingle size.
    signature_size = 372            # Size of the signature matrix. Approximately 50% of the number of rows of the binary vector representations.
    data, dataframe = get_data()
    most_frequent_keys, keyvalues_dataframe = find_most_common_keyvalues(data, number_of_keys)
    clean_dataframe = clean_data(dataframe, keyvalues_dataframe, most_frequent_keys)
    length = len(clean_dataframe)

    total_products_considered = 0
    result = pd.DataFrame(columns=['Bootstrap', 'PQ LSH', 'PC LSH', 'F1-star', 'Fraction comparisons LSH', 'PQ MSM', 'PC MSM', 'F1', 'Fraction comparisons MSM', 'b', 'r', 'Threshold'])
    for bootstrap in range(number_of_bootstraps):
        index = 0
        bootstrap_dataframe = resample(clean_dataframe, n_samples=math.floor(0.6*length), replace=False, random_state=bootstrap)
        bootstrap_dataframe.reset_index(inplace=True)

        percentage_products_considered = len(bootstrap_dataframe) / len(clean_dataframe)
        total_products_considered += len(bootstrap_dataframe)

        binary_vector_representations = create_binary_vector_representations(bootstrap_dataframe, most_frequent_keys)
        signature_matrix = min_hashing(signature_size, binary_vector_representations)
        bands, rows = possible_values_br(signature_matrix)

        for i in range(len(bands)):
            number_of_duplicates = count_duplicates(bootstrap_dataframe)
            pairs_lsh = locality_sensitive_hashing(signature_matrix, bands[i], rows[i])
            pair_quality_lsh, pair_completeness_lsh, f1_star, fraction_of_comparisons_lsh = evaluate_performance(pairs_lsh, bootstrap_dataframe)

            dissimilarity_matrix = create_dissimilarity_matrix(bootstrap_dataframe, pairs_lsh, k, most_frequent_keys)
            threshold = optimal_threshold(dissimilarity_matrix, bootstrap_dataframe)
            pairs_clustering = clustering(dissimilarity_matrix, threshold)
            pair_quality_clustering, pair_completeness_clustering, f1, fraction_of_comparisons_clustering = evaluate_performance(pairs_clustering, bootstrap_dataframe)

            result.loc[index, ['Bootstrap']] = bootstrap + 1
            result.loc[index, ['PQ LSH']] = pair_quality_lsh
            result.loc[index, ['PC LSH']] = pair_completeness_lsh
            result.loc[index, ['F1-star']] = f1_star
            result.loc[index, ['Fraction comparisons LSH']] = fraction_of_comparisons_lsh
            result.loc[index, ['PQ MSM']] = pair_quality_clustering
            result.loc[index, ['PC MSM']] = pair_completeness_clustering
            result.loc[index, ['F1']] = f1
            result.loc[index, ['Fraction comparisons MSM']] = fraction_of_comparisons_clustering
            result.loc[index, ['b']] = bands[i]
            result.loc[index, ['r']] = rows[i]
            result.loc[index, ['Threshold']] = threshold

            index += 1

            print(f'Result of combination {i + 1} (out of {len(bands)} possible combinations) of bootstrap {bootstrap + 1} (of {number_of_bootstraps} total bootstraps):')
            print(result.iloc[index - 1])

        result.to_excel("C:\\prive\\jaco ckv\\Master Econometrics and Management Science\\Blok 2\\Computer Science\\Assignment\\python\\Duplicate Detection\\Duplicate-Detection\\Results\\result_bootstrapping_bootstrap" + str(bootstrap + 1) + "_keys" + str(number_of_keys) + ".xlsx")

    average_percentage_products_considered = (total_products_considered / len(clean_dataframe)) / number_of_bootstraps
    print(average_percentage_products_considered)
