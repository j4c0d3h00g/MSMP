from data_preprocessing import find_modelword, get_k_shingle, get_data, clean_data
import numpy as np


def create_binary_vector_representations(shingle_size, dataframe):
    n_cols = len(dataframe)
    all_modelwords = []
    all_unique_words = []

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

    # print(all_modelwords)
    print(all_unique_words)
    print(binary_vector_representations)


if __name__ == "__main__":
    _, dataframe = get_data()
    dataframe = clean_data(dataframe)
    create_binary_vector_representations(4, dataframe)







