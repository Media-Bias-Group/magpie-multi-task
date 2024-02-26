"""This module contains the Preprocessor for the 9 BASIL dataset."""

import itertools
import json
import os
from typing import List, Tuple

import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING_BIAS = {"informational": 1, "lexical": 0, "neutral": 2}
MAPPING_AIM = {"direct": 1, "indirect": 0, "neutral": 2}


class Preprocessor09Basil(PreprocessorBlueprint):
    """Preprocessor for the 09_BASIL dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor09Basil."""
        super(Preprocessor09Basil, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Tuple[List, List], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 09_BASIL."""
        article_data, annotation_data = raw_data
        biased_observations = []
        non_biased_observations = []
        for i, art in enumerate(article_data):
            ann = annotation_data[i]

            paragraphs = art.get("body-paragraphs")  # Now a list of paragraphs
            annotations = ann.get("phrase-level-annotations")  # Now a list of annotations

            phrases = list(itertools.chain.from_iterable(paragraphs))

            biased_sent_ids = []
            for ann in annotations:
                # The id can be in 2 different formats:
                # "p<prase-id>" or "title"
                id = ann["id"]
                text = ann["txt"]
                bias = 1 if ann["bias"] == "inf" else 0  # Type of bias. (1=informational, 0=lexical)

                aim = (
                    1 if ann["aim"] == "dir" else 0
                )  # direct/ indirect. If indirect, annotations for ...-sentiment are available.
                if id == "title":
                    phrase = art.get("title")
                else:
                    id = int(id.split("p")[-1])
                    phrase = phrases[id]
                    biased_sent_ids.append(id)
                assert text in phrase

                biased_observation = {"text": phrase, "pos": text, "label": bias, "aim": aim}
                biased_observations.append(biased_observation)

            non_biased_sents = set(range(len(phrases))) - set(biased_sent_ids)
            non_biased_observations.extend(
                [{"text": phrases[i], "pos": None, "label": 2, "aim": 2} for i in non_biased_sents]
            )
        biased_df = pd.DataFrame(biased_observations)
        non_biased_df = pd.DataFrame(non_biased_observations)
        df = pd.concat([biased_df, non_biased_df], axis=0)

        self._log_before_preprocessing(data=df)

        df["pos"] = df["pos"].fillna("")
        df["pos"] = df["pos"].apply(lambda x: self._unify_text(text=x, rm_hashtag=False))

        df = self._clean(df=df, length_threshold=length_threshold, rm_hashtag=False)
        self._log_after_preprocessing(data=df)

        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "aim_value_counts": df.aim.value_counts().to_dict(),
                "MAPPING_BIAS": MAPPING_BIAS,
                "MAPPING_AIM": MAPPING_AIM,
            },
        )
        return df

    def _load_raw_data_from_local(self) -> Tuple[List, List]:
        """Load the raw data of 09_BASIL."""
        articles, annotations = [], []
        for year in range(2010, 2020):
            arts = sorted(os.listdir(os.path.join(self._raw_data_local_path, "articles", str(year))))

            anns = sorted(os.listdir(os.path.join(self._raw_data_local_path, "annotations", str(year))))
            anns_cut = ["".join(ann.split("_ann")) for ann in anns]

            assert arts == anns_cut

            for i, art in enumerate(arts):
                try:
                    with open(os.path.join(self._raw_data_local_path, "articles", str(year), art), "r") as f:
                        article_data = json.load(f)

                    with open(os.path.join(self._raw_data_local_path, "annotations", str(year), anns[i]), "r") as f:
                        annotation_data = json.load(f)

                    articles.append(article_data)
                    annotations.append(annotation_data)
                except json.decoder.JSONDecodeError:
                    print("Caught error. Attempted to load an empty file.")

        return articles, annotations
