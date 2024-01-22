import re
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import ray
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

nltk.download("stopwords")
STOPWORDS = nltk.corpus.stopwords.words("english")


def load_data(dataset_loc: str, num_samples: int = None) -> Dataset:
    """
    Load and shuffle a Ray Dataset.

    Args:
        dataset_loc (str): Location of the CSV dataset file
        num_samples (int): Number of samples to load. Defaults to None

    Returns:
        Dataset: Loaded and shuffled dataset
    """
    ds = ray.data.read_csv(dataset_loc)
    ds = ds.random_shuffle(seed=42)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds


def preprocess_text(text: str, stopwords: List = STOPWORDS) -> str:
    """
    Preprocess raw text.

    Args:
        text (str): Raw text to clean
        stopwords (List, optional): List of stopwords to filter. Defaults to STOPWORDS

    Returns:
        str: Cleaned text
    """
    text = text.lower()
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text)
    text = re.sub(" +", " ", text)
    text = text.strip()
    text = re.sub(r"http\S+", "", text)

    return text


def tokenize(batch: Dict) -> Dict:
    """
    Tokenize the text input in batch using a Bert Tokenizer.

    Args:
        batch (Dict): Batch of data with the text inputs to tokenize

    Returns:
        Dict: Tokenization results on the text inputs
    """
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["category"]))


def preprocess_df(df: pd.DataFrame, class_to_index: Dict) -> Dict:
    """
    Preprocess the data in DataFrame.

    Args:
        df (pd.DataFrame): Raw DataFrame to preprocess
        class_to_index (Dict): A dictionary mapping class names to their corresponding indices

    Returns:
        Dict: Preprocessed results. (ids, masks, targets)
    """
    df["text"] = df.headline + " " + df.keywords
    df["text"] = df.text.apply(preprocess_text)
    df = df.drop(columns=["headline", "links", "short_description", "keywords"], errors="ignore")
    df = df[["text", "category"]]
    df["category"] = df["category"].map(class_to_index)
    outputs = tokenize(df)
    return outputs


def stratify_split(ds: Dataset, stratify: str, test_size: float, shuffle: bool = True, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into training and testing sets,
    to make sure an equal distribution of data points from each class.

    Args:
        ds (Dataset): Ray Dataset to split
        stratify (str): Split on column name
        test_size (float): Percentage of the dataset allocated for the test set
        shuffle (bool, optional): Whether to randomly shuffle the dataset. Defaults to True
        seed (int, optional): Randomization seed for shuffling. Defaults to 42

    Returns:
        Tuple[Dataset, Dataset]: The train and test datasets, both stratified
    """

    def _add_split(df: pd.DataFrame) -> pd.DataFrame:
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
        return df[df["_split"] == split].drop("_split", axis=1)

    grouped = ds.groupby(stratify).map_groups(_add_split, batch_format="pandas")
    train_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "train"}, batch_format="pandas")
    test_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "test"}, batch_format="pandas")

    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds


class Preprocessor:
    """
    This class is designed to handle various data preprocessing tasks.

    Parameters:
        class_to_index (dict, optional): An optional initial mapping of class names to indices. Defaults to an empty dictionary

    Methods:
        fit(ds): Update the class-to-index mapping
        transform(ds): Apply preprocessing to the Ray Dataset
    """

    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def fit(self, ds):
        categories = ds.unique(column="category")
        self.class_to_index = {tag: i for i, tag in enumerate(categories)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self

    def transform(self, ds):
        return ds.map_batches(preprocess_df, fn_kwargs={"class_to_index": self.class_to_index}, batch_format="pandas")
