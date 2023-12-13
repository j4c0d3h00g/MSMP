from data_preprocessing import *
import collections
import itertools
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.cluster import AgglomerativeClustering
from random import randint
from sympy import randprime



def create_binary_vector_representations(dataframe, most_frequent_keys):
    n_cols = len(dataframe)
    all_modelwords_title = []
    all_modelwords_keyvalues = []
    all_unique_words = []

    for i in range(n_cols):
        all_modelwords_title.append(find_modelwords_title(dataframe['title'][i]))       # Find model words in title.

    for i in range(n_cols):
        modelwords_keyvaluepairs = []
        for j in range(len(most_frequent_keys)):
            if dataframe[most_frequent_keys[j]][i] is not None:
                modelwords = find_modelwords_keyvaluepairs(dataframe[most_frequent_keys[j]][i])     # Find model words in key-value pairs.
                modelwords_keyvaluepairs = list(itertools.chain(modelwords_keyvaluepairs, modelwords))

        all_modelwords_keyvalues.append(modelwords_keyvaluepairs)

    all_modelwords = list(itertools.chain(all_modelwords_title, all_modelwords_keyvalues))      # Create a list containing all model words.

    # Find all unique model words in the list containing all model words.
    for i in range(len(all_modelwords)):
        words = get_k_shingles(all_modelwords[i], 1)
        for word in words:
            if word not in all_unique_words:
                all_unique_words.append(word)

    n_unique_words = len(all_unique_words)
    binary_vector_representations = np.zeros((n_unique_words, n_cols))
    for i in range(n_unique_words):
        for j in range(n_cols):
            if set(all_unique_words[i]).issubset(set(all_modelwords_title[j])) or set(all_unique_words[i]).issubset(set(all_modelwords_keyvalues[j])):
                binary_vector_representations[i, j] = 1

    return binary_vector_representations


def min_hashing(signature_size, binary_vector_representations):
    n_modelwords, n_products = binary_vector_representations.shape
    k = signature_size

    signature_matrix = np.full((k, n_products), np.inf)

    a = np.zeros(k)
    b = np.zeros(k)
    p = np.zeros(k)
    for i in range(k):
        a[i] = randint(1, n_modelwords)
        b[i] = randint(1, n_modelwords)
        p[i] = randprime(k + 1, (3 * (k + 1)) + 1)

    random_hash_functions = np.zeros(k)
    for row in range(n_modelwords):
        for i in range(k):
            random_hash_functions[i] = (a[i] + b[i] * row) % p[i]

        for column in range(n_products):
            if binary_vector_representations[row, column] == 1:
                minimum_values = np.minimum(signature_matrix[:, column], random_hash_functions)
                signature_matrix[:, column] = minimum_values

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


# Additional method to find candidate pairs based on matching values of the most occurring keys.
# This method is not used in my research.
def find_pairs_keyvalues(dataframe, most_frequent_keys, similarity_threshold):
    potential_pairs = []

    for i in range(len(dataframe)):
        for j in range(i + 1, len(dataframe)):
            number_of_similarities = 0

            product1 = i
            product2 = j
            for keyvalue_pair in range(len(most_frequent_keys)):
                if dataframe[most_frequent_keys[keyvalue_pair]][i] is not None and dataframe[most_frequent_keys[keyvalue_pair]][j] is not None and dataframe[most_frequent_keys[keyvalue_pair]][i] == dataframe[most_frequent_keys[keyvalue_pair]][j]:
                    number_of_similarities += 1
                    if number_of_similarities >= similarity_threshold:
                        pair_tuple = tuple((product1, product2))
                        if dataframe['shop'][product1] != dataframe['shop'][product2]:
                            potential_pairs.append(pair_tuple)

                        continue

    potential_pairs_set = set(potential_pairs)

    return potential_pairs_set


def create_dissimilarity_matrix(dataframe, potential_pairs, k, most_frequent_keys):
    n_products = len(dataframe)
    dissimilarity_matrix = np.full((n_products, n_products), 100, float)
    np.fill_diagonal(dissimilarity_matrix, 0)

    for pair in potential_pairs:
        product1 = pair[0]
        product2 = pair[1]

        shingles1 = get_k_shingles(dataframe['title'][product1], k)     # Derive k-shingles in title of product 1.
        shingles2 = get_k_shingles(dataframe['title'][product2], k)     # Derive k-shingles in title of product 2.

        for i in range(len(most_frequent_keys)):
            if dataframe[most_frequent_keys[i]][product1] is not None:
                # Derive k-shingles in key-value pairs of product 1.
                shingles1.extend(get_k_shingles(dataframe[most_frequent_keys[i]][product1], k))
            if dataframe[most_frequent_keys[i]][product2] is not None:
                # Derive k-shingles in key-value pairs of product 2.
                shingles2.extend(get_k_shingles(dataframe[most_frequent_keys[i]][product2], k))

        set_shingles1 = set(shingles1)
        set_shingles2 = set(shingles2)
        similarity = jaccard_similarity(set_shingles1, set_shingles2)       # Compute Jaccard similarity between the two products.
        dissimilarity_matrix[product1, product2] = 1 - similarity

        if dataframe['shop'][product1] == dataframe['shop'][product2]:
            dissimilarity_matrix[product1, product2] = 100

        if dataframe['brand'][product1] is not None and dataframe['brand'][product2] is not None and dataframe['brand'][product1] != dataframe['brand'][product2]:
            dissimilarity_matrix[product1, product2] = 100

        dissimilarity_matrix[product2, product1] = dissimilarity_matrix[product1, product2]

    return dissimilarity_matrix


def clustering(dissimilarity_matrix, threshold):
    linkage_clustering = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold)
    clusters = linkage_clustering.fit_predict(dissimilarity_matrix)

    n_clusters = len(clusters)
    cluster_dict = collections.defaultdict(set)
    potential_pairs = []

    for index in range(n_clusters):
        cluster_dict[clusters[index]].add(index)

    for potential_pair in cluster_dict.values():
        if len(potential_pair) > 1:
            for pair in itertools.combinations(potential_pair, 2):
                potential_pairs.append(pair)

    potential_pairs = [element for element in potential_pairs if element[0] < element[1]]
    potential_pairs_set = set(potential_pairs)

    return potential_pairs_set
