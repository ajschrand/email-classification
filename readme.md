
# Python Setup Instructions for Enron E-mail Classification Project

## Prerequisites

- Python 3 in path (assumed to be named `python`)
- `pip` installed

## Steps

1. Install requirements:

    ```py
    pip install -r requirements.txt
    ```

1. Download the parsed dataset from [https://drive.google.com/file/d/1XESP_gUzaxc0VZSTgVVMkMfNa5GmLtTD/view?usp=sharing](https://drive.google.com/file/d/1XESP_gUzaxc0VZSTgVVMkMfNa5GmLtTD/view?usp=sharing). Place it in the same directory as the code.
    - It is a preprocessed version of the original, found at [https://www.cs.cmu.edu/~enron/](https://www.cs.cmu.edu/~enron/), to save time. If you would like to parse the data yourself, extract it to the same directory as the code and run

        ```py
        python parser.py
        ```

1. To clean the dataset, run

    ```py
    python clean_parsed_emails.py
    ```

1. To generate features for decision tree, run

    ```py
    python feature_generator.py
    ```

1. To generate decision trees, run

    ```py
    python decision_tree.py
    ```

1. To generate features for k-nearest neighbor, run

    ```py
    python BOW_generator.py
    ```

1. To generate k-nearest neighbor models, run

    ```py
    python KNN.py
    ```
