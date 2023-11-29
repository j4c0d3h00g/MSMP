from data_preprocessing import *
from functions import *
from performance_evaluation import *
from sklearn.utils import resample
import pandas as pd


if __name__ == "__main__":
    number_of_bootstraps = 5
    k = 4
    _, dataframe = get_data()
    clean_dataframe = clean_data(dataframe)

    total_products_considered = 0

    result = pd.DataFrame(columns=['Bootstrap', 'F1-star', 'Fraction comparisons LSH', 'PQ', 'PC', 'F1', 'Fraction comparisons MSM', 'b', 'r', 'Threshold'])
    for bootstrap in range(number_of_bootstraps):
        bootstrap_dataframe = resample(clean_dataframe, replace=True, random_state=42)
        bootstrap_dataframe.drop_duplicates(inplace=True)
        bootstrap_dataframe.reset_index(inplace=True)

        percentage_products_considered = len(bootstrap_dataframe) / len(clean_dataframe)
        total_products_considered += len(bootstrap_dataframe)

        binary_vector_representations = create_binary_vector_representations(bootstrap_dataframe)
        signature_matrix = min_hashing(0.5, binary_vector_representations)
        bands, rows = possible_values_br(signature_matrix)

        for i in range(len(bands)):
            pairs_lsh = locality_sensitive_hashing(signature_matrix, bands[i], rows[i])
            _, _, f1_star, fraction_of_comparisons_lsh = evaluate_performance(pairs_lsh, bootstrap_dataframe)

            dissimilarity_matrix = create_dissimilarity_matrix(bootstrap_dataframe, pairs_lsh, 4)
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

    average_percentage_products_considered = (total_products_considered / len(clean_dataframe)) / number_of_bootstraps
    print(average_percentage_products_considered)
    result.to_excel("result_bootstrapping.xlsx")

