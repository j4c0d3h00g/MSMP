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
    dataframe = pd.DataFrame(dataframe, columns=['modelID', 'title', 'shop', 'brand'])

    # need to remove capital  letters, interpunction etc from title, herz=hz, inch, remove store name, remove weird notation
    dataframe['title'] = dataframe['title'].str.lower()     # remove capital letters
    dataframe['title'] = dataframe['title'].str.replace("-", "")
    dataframe['title'] = dataframe['title'].str.replace("/", "")
    dataframe['title'] = dataframe['title'].str.replace("(", "")
    dataframe['title'] = dataframe['title'].str.replace(")", "")
    dataframe['title'] = dataframe['title'].str.replace("[", "")
    dataframe['title'] = dataframe['title'].str.replace("]", "")
    dataframe['title'] = dataframe['title'].str.replace("|", "")
    dataframe['title'] = dataframe['title'].str.replace("+", "")
    dataframe['title'] = dataframe['title'].str.replace(":", "")
    dataframe['title'] = dataframe['title'].str.replace("  ", " ")

    dataframe['title'] = dataframe['title'].str.replace("inches", "inch")
    dataframe['title'] = dataframe['title'].str.replace("\"", "inch")
    dataframe['title'] = dataframe['title'].str.replace(" inch", "inch")
    dataframe['title'] = dataframe['title'].str.replace("'", "inch")
    dataframe['title'] = dataframe['title'].str.replace("''", "inch")

    dataframe['title'] = dataframe['title'].str.replace("hertz", "hz")
    dataframe['title'] = dataframe['title'].str.replace(" hz", "hz")

    dataframe['title'] = dataframe['title'].str.replace("amazon.com", "")
    dataframe['title'] = dataframe['title'].str.replace("bestbuy.com", "")
    dataframe['title'] = dataframe['title'].str.replace("best buy", "")
    dataframe['title'] = dataframe['title'].str.replace("newegg.com", "")
    dataframe['title'] = dataframe['title'].str.replace("thenerds.net", "")

    dataframe['title'] = dataframe['title'].str.strip()

    dataframe['brand'] = dataframe['brand'].str.lower()     # remove capital letters
    dataframe['shop'] = dataframe['shop'].str.lower()   # remove capital letters
    return dataframe


def find_modelword(title):
    regex = re.compile(r'(\b(?:[0-9]*)[a-z]+\b)|(\b[a-z]+\b)')
    modelword = [x for sublist in regex.findall(title) for x in sublist if x != ""]
    return modelword


def get_k_shingle(text, k):
    shingles = []
    for i in range(0, len(text) - k + 1):
        shingles.append(text[i:i+k])

    return shingles


if __name__ == "__main__":
    data, dataframe = get_data()
    print(dataframe)
    clean_data(dataframe)
    # print(data)
    # print(len(data))
    print(count_products(data))
