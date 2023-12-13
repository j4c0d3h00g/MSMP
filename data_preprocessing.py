import json
import re
import pandas as pd


def get_data():
    path = 'C:\\prive\\jaco ckv\\Master Econometrics and Management Science\\Blok 2\\Computer Science\\Assignment\\TVs-all-merged\\TVs-all-merged.json'

    f = open(path)
    data = json.load(f)

    dataframe = []
    for key, values in data.items():
        for product in values:
            dataframe.append((key, product.get('title'), product.get('shop'), product.get('featuresMap').get('Brand')))

    return data, dataframe


def count_products(data):
    number_of_products = 0
    for key in data.keys():
        number_of_products += len(data[key])

    return number_of_products


def find_most_common_keyvalues(data, n_keyvalues):
    frequency_keyvalues = {}

    for _, values in data.items():
        for product in values:
            for key in product.get('featuresMap'):
                if key not in frequency_keyvalues:
                    frequency_keyvalues[key] = 1
                else:
                    frequency_keyvalues[key] += 1

    frequency_keyvalues.pop('Brand')
    frequency_keyvalues.pop('UPC')
    most_frequent_keys = sorted(frequency_keyvalues, key=frequency_keyvalues.get, reverse=True)[:n_keyvalues]

    keyvalues_dataframe = []
    for key, values in data.items():
        for product in values:
            product_values = [None] * len(most_frequent_keys)
            for i in range(len(most_frequent_keys)):
                product_values[i] = product.get('featuresMap').get(most_frequent_keys[i])

            keyvalues_dataframe.append(product_values)

    return most_frequent_keys, keyvalues_dataframe


def clean_data(dataframe, keyvalues_dataframe, most_frequent_keys):
    column_names = ['modelID', 'title', 'shop', 'brand']
    pd_dataframe = pd.DataFrame(dataframe, columns=column_names)
    pd_keyvalues_dataframe = pd.DataFrame(keyvalues_dataframe, columns=most_frequent_keys)

    clean_dataframe = pd.concat([pd_dataframe, pd_keyvalues_dataframe], axis=1)

    unique_brands = []
    for i in range(len(clean_dataframe)):
        if clean_dataframe['brand'][i] not in unique_brands and clean_dataframe['brand'][i] is not None:
            unique_brands.append(clean_dataframe['brand'][i])

    for i in range(len(clean_dataframe)):
        if clean_dataframe['brand'][i] is None:
            brand_name = list(filter(lambda brand: brand in clean_dataframe['title'][i], unique_brands))
            if brand_name:
                clean_dataframe['brand'][i] = brand_name[0]

    inch_list = ["inches", "\"", " inch", "'", "''", "”"]
    hertz_list = ["hertz", " hz"]
    site_list = ["amazon.com", "bestbuy.com", "best buy", "newegg.com", "thenerds.net"]

    clean_dataframe['title'] = clean_dataframe['title'].str.lower()     # Remove capital letters.
    for i in range(len(clean_dataframe)):
        clean_dataframe['title'][i] = re.sub("[^a-zA-Z0-9\s\.]", "", clean_dataframe['title'][i])

        for inch in inch_list:
            clean_dataframe['title'][i] = clean_dataframe['title'][i].replace(inch, "inch")

        for hertz in hertz_list:
            clean_dataframe['title'][i] = clean_dataframe['title'][i].replace(hertz, "hz")

        for site in site_list:
            clean_dataframe['title'][i] = clean_dataframe['title'][i].replace(site, "")

    clean_dataframe['title'] = clean_dataframe['title'].str.strip()     # Remove redundant white space.

    clean_dataframe['brand'] = clean_dataframe['brand'].str.lower()     # Remove capital letters.
    clean_dataframe['shop'] = clean_dataframe['shop'].str.lower()       # Remove capital letters.

    for i in range(len(most_frequent_keys)):
        clean_dataframe[most_frequent_keys[i]] = clean_dataframe[most_frequent_keys[i]].str.lower()
        for j in range(len(clean_dataframe)):
            if clean_dataframe[most_frequent_keys[i]][j] is not None:
                # Remove special characters except ., -, and :.
                clean_dataframe[most_frequent_keys[i]][j] = re.sub("[^a-zA-Z0-9\.\-\:]", " ", clean_dataframe[most_frequent_keys[i]][j])

    return clean_dataframe


def find_modelwords_title(expression):
    regex = re.compile(r'(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:([0-9]+\.[0-9]+)[^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))')
    modelwords = [x for sublist in regex.findall(expression) for x in sublist if x != ""]
    return modelwords


def find_modelwords_keyvaluepairs(expression):
    regex = re.compile(r'(?:^\d*\.?\d*$)|(?:^\d*\:?\d*$)|(?:^\d*\-?\d*$)')
    modelwords = [x for sublist in regex.findall(expression) for x in sublist if x != ""]
    return modelwords


def get_k_shingles(text, k):
    shingles = []
    for i in range(0, len(text) - k + 1):
        shingles.append(text[i:i+k])

    return shingles


def possible_values_br(signature_matrix):
    n, _ = signature_matrix.shape
    bands = []
    rows = []

    for b in range(1, n + 1):
        for r in range(1, n + 1):
            if b * r == n:
                bands.append(b)
                rows.append(r)

    return bands, rows


def jaccard_similarity(shingle1, shingle2):
    similarity = len(shingle1.intersection(shingle2)) / len(shingle1.union(shingle2))
    return similarity


def count_duplicates(dataframe):
    number_of_duplicates = 0

    for i in range(len(dataframe)):
        for j in range(i + 1, len(dataframe)):
            if dataframe['modelID'][i] == dataframe['modelID'][j]:
                number_of_duplicates += 1

    return number_of_duplicates

