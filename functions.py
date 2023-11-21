from data_preprocessing import find_modelword, get_k_shingle, get_data, clean_data, possible_values_br, jaccard_similarity
import numpy as np
import pandas as pd
import math
import collections
import itertools
import random



def create_binary_vector_representations(dataframe):
    n_cols = len(dataframe)
    all_modelwords = []
    all_unique_words = []

    # to-do: modelwords in key-value pairs
    for i in range(n_cols):
        all_modelwords.append(find_modelword(dataframe['title'][i]))    # Nested list where list i contains the modelwords of product i
    for i in range(len(all_modelwords)):
        words = get_k_shingle(all_modelwords[i], 1)
        for word in words:
            if word not in all_unique_words:
                all_unique_words.append(word)

    n_unique_words = len(all_unique_words)
    binary_vector_representations = np.zeros((n_unique_words, n_cols))
    for i in range(n_unique_words):
        for j in range(n_cols):
            binary_vector_representations[i, j] = set(all_unique_words[i]).issubset(all_modelwords[j])

    return binary_vector_representations


def min_hashing(multiplication_factor, binary_vector_representations):
    n_modelwords, n_products = binary_vector_representations.shape
    binary_vector_representations_frame = pd.DataFrame(binary_vector_representations, columns=np.arange(n_products))
    k = math.ceil(multiplication_factor * n_modelwords)

    signature_matrix = np.zeros((k, n_products))
    for i in range(k):
        binary_vector_representations_frame_permuted = binary_vector_representations_frame.sample(frac=1)

        for j in range(n_products):
            column = binary_vector_representations_frame_permuted.iloc[:, j]
            row_index = (column.values == 1).argmax()
            signature_matrix[i, j] = row_index

    return signature_matrix


def locality_sensitive_hashing(signature_matrix, b, r):
    n, n_products = signature_matrix.shape
    assert (n % b == 0)
    assert (b * r == n)

    bands = np.split(signature_matrix, b, axis=0)
    potential_pairs = []

    buckets = collections.defaultdict(set)
    for item, band in enumerate(bands):
        for product in range(n_products):
            band_index = tuple(list(band[:, product]) + [str(item)])
            buckets[band_index].add(product)

    for potential_pair in buckets.values():
        if len(potential_pair) > 1:
            for pair in itertools.combinations(potential_pair, 2):
                potential_pairs.append(pair)

    potential_pairs = [element for element in potential_pairs if element[0] < element[1]]
    potential_pairs_set = set(potential_pairs)

    return potential_pairs_set


def create_dissimilarity_matrix(dataframe, potential_pairs, k):
    n_products = len(dataframe)
    dissimilarity_matrix = np.full((n_products, n_products), 100, float)
    np.fill_diagonal(dissimilarity_matrix, 0)

    for pair in potential_pairs:
        product1 = potential_pairs[0]
        product2 = potential_pairs[1]

        shingle1 = get_k_shingle(dataframe['title'][product1], k)
        shingle2 = get_k_shingle(dataframe['title'][product2], k)

        dissimilarity_matrix[product1, product2] = 1 - jaccard_similarity(set(shingle1), set(shingle2))






if __name__ == "__main__":
    _, dataframe = get_data()
    dataframe = clean_data(dataframe)
    binary_vector_representations = create_binary_vector_representations(dataframe)
    signature_matrix = min_hashing(0.5, binary_vector_representations)
    pairs = locality_sensitive_hashing(signature_matrix, 94, 6)








