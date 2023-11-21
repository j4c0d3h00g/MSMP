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


def clean_data(dataframe):
    clean_dataframe = pd.DataFrame(dataframe, columns=['modelID', 'title', 'shop', 'brand'])

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

    clean_dataframe['title'] = clean_dataframe['title'].str.lower()     # remove capital letters
    for i in range(len(clean_dataframe)):
        clean_dataframe['title'][i] = re.sub("[^a-zA-Z0-9\s\.]", "", clean_dataframe['title'][i])

        for inch in inch_list:
            clean_dataframe['title'][i] = clean_dataframe['title'][i].replace(inch, "inch")

        for hertz in hertz_list:
            clean_dataframe['title'][i] = clean_dataframe['title'][i].replace(hertz, "hz")

        for site in site_list:
            clean_dataframe['title'][i] = clean_dataframe['title'][i].replace(site, "")

    clean_dataframe['title'] = clean_dataframe['title'].str.strip()

    clean_dataframe['brand'] = clean_dataframe['brand'].str.lower()     # remove capital letters
    clean_dataframe['shop'] = clean_dataframe['shop'].str.lower()   # remove capital letters
    return clean_dataframe


def find_modelword(title):
    regex = re.compile(r'(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:([0-9]+\.[0-9]+)[^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))')
    modelword = [x for sublist in regex.findall(title) for x in sublist if x != ""]
    return modelword


def get_k_shingle(text, k):
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


if __name__ == "__main__":
    data, dataframe = get_data()
    print(dataframe)
    clean_dataframe = clean_data(dataframe)
    print(clean_dataframe)
    # print(data)
    # print(len(data))
    print(count_products(data))
