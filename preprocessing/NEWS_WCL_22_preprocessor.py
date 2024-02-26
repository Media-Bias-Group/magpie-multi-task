"""This module contains the Preprocessor for the 22 NEWS WCL dataset."""

from typing import Tuple

import nltk
import pandas as pd
from newspaper import Article, ArticleException

nltk.download("punkt")
import os

import numpy as np
from nltk import sent_tokenize
from tqdm import tqdm

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = dict(zip(["LL", "L_", "M_", "R_", "RR"], np.linspace(0, 1, 5)))


class Preprocessor22NewsWCL(PreprocessorBlueprint):
    """Preprocessor for the 22 News WCL 50 dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor12NewsWCL."""
        super(Preprocessor22NewsWCL, self).__init__(*args, **kwargs)

    def _get_article_text_from_url(self, url: str) -> str:
        """Fetch article text from URL. If the url is unavailable or the text is unparsable, empty string is returned."""
        article = Article(url)
        try:
            article.download()
            article.parse()
        except ArticleException:
            return ""

        return str(article.text)

    def _extract_sentence(self, article_text, group_list):
        """Find the sentence from which the phrases in group_list were drawn.

        Args:
            article_text (_type_): article text
            group_list (_type_): list of phrases from the same sentence
        Returns:
            str : cleaned sentence or None if not found
        """
        # split article to sentences
        sentences = sent_tokenize(str(article_text))

        if not sentences:
            return None

        for sent in sentences:
            match = 0
            for phrase in group_list:
                if phrase in sent:
                    match += 1

            if match > 1 or len(group_list) == 1 and match == 1:
                # clean sentence
                sent = sent.replace("Advertisement", "")
                sent = sent.replace("\n\n", "")
                return sent

        return None

    def _preprocess(self, raw_data: Tuple[pd.DataFrame, pd.DataFrame], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 22NewsWCL."""
        tqdm.pandas()
        anns, urls = raw_data

        self._log_before_preprocessing(data=anns)

        # construct IDs
        urls["ID"] = urls["Event ID"].astype(str) + "_" + urls["Outlet"]
        urls["ID"] = urls["ID"].apply(lambda x: x + "_" if len(x) == 3 else x)
        urls.drop(["Event ID", "Outlet"], axis=1, inplace=True)
        # download texts
        urls["article"] = urls["URL"].apply(self._get_article_text_from_url)

        # construct IDs
        anns["ID"] = anns["id"].apply(lambda x: str(x)[:4])
        anns.drop(["code_type", "code_name", "target_concept", "event_id", "publisher_id"], axis=1, inplace=True)

        urls = urls.reset_index()  # make sure indexes pair with number of rows

        labels = []
        sentences = []

        # go through the annotations that mark the same sentence and find this sentence
        for name, group in anns.groupby(["ID", "paragraph", "sentence"]):
            parts = group["code_mention"].to_list()
            text = urls[urls["ID"] == name[0]]["article"].item()  # get the text
            s = self._extract_sentence(text, parts)

            if s is not None and s not in sentences:
                labels.append(group["ID"].iloc[0][-2:])
                sentences.append(s)

        data = pd.DataFrame({"text": sentences, "label": labels})
        data["label"] = data.label.map(MAPPING)

        cleaned = self._clean(data, length_threshold=length_threshold)
        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the raw data of 22_NewsWCL."""
        anns = pd.read_csv(
            os.path.join(self._raw_data_local_path, "Annotations.csv"), skiprows=[0]
        )  # first row is corrupted
        urls = pd.read_csv(os.path.join(self._raw_data_local_path, "urls.tsv"), sep="\t")

        return anns, urls
