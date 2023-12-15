# Multi-component Similarity Method with Pre-selection+
This repository contains the implementation of the Multi-component Similarity Method with Pre-selection+ (MSMP+). In this implementation of the MSMP+ algorithm it is possible to adjust the amount of information from the key-value pairs that is added in the product description. 

## Data
The data is contained in the file `TVs-all-merged.json`. This dataset consists of televisions that are sold by the Web shops www.amazon.com, www.bestbuy.com, www.thenerds.net, and www.newegg.com. This dataset consists of 1624 products, where each product has a title together with additional information that is contained in the key-value pairs.

## Files
This repository contains the following Python files:
- `data_preprocessing.py`: Contains methods to retrieve and clean the data. In this file, also other functional methods for e.g. finding model words and creating shingles of size k are included. 
- `functions.py`: Contains the functionality of the MSMP+ algorithm. This file consists of the methods that create binary vector representations for the products, perform Locality Sensitive Hashing (LSH), and apply a more general form of the Multi-component Similarity Method (MSM).
- `performance_evaluation.py`: Contains a method to derive the performance evaluation measures and a method to derive the optimal threshold for the adapted hierarchical single linkage clustering that is part of MSM.
- `bootstrapping.py`: Performs bootstrapping in order to consistently evaluate the performance of MSMP+ for various amounts of added information from the key-value pairs.
- `plots.py`: Plots the average performance measures across the number of bootstraps against the fraction of comparisons made.

## Usage
In order to obtain the performance evaluation results of MSMP+ where the amount of added information from the key-value pairs is varied, one needs to run `bootstrapping.py`. In this file, the amount of added information from the key-value pairs and the shingle size can be adjusted. Running this file returns xlsx files for each bootstrap. After obtaining the xlsx files, one can run the file `plots.py` to obtain plots of the average performance measures across the number of bootstraps against the fraction of comparisons made. Note that the save location for storing the xlsx files needs to be adjusted to your own save location in order to run the files `bootstrapping.py` and `plots.py`. 

## Acknowledgements
The implementation of this repository is based on the description of the MSMP+ algorithm in the research of:
Hartveld, A., van Keulen, M., Mathol, D., van Noort, T., Plaatsman, T., Frasincar, F., & Schouten, K.: An LSH-based model-words-driven product duplicate detection method. In: 30th International Conference on Advanced Information Systems Engineering. vol. 10816, pp. 409-423. Springer (2018).
