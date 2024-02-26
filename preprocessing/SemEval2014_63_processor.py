"""This module contains the Preprocessor for the 63 SemEval2014 dataset."""

import os
from collections import OrderedDict
from typing import List, Tuple

import pandas as pd
import xmltodict

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"neutral": 0, "negative": 1, "positive": 3}


class Preprocessor63SemEval2014(PreprocessorBlueprint):
    """Preprocessor for the 63 SemEval2014 dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor 63 SemEval2014."""
        super(Preprocessor63SemEval2014, self).__init__(*args, **kwargs)
        self._processed_data_local_path = os.path.join(kwargs["local_path"], "preprocessed.csv")
        self._processed_data_gcs_path = os.path.join(kwargs["local_path"], "preprocessed.csv")

    def _cast_terms_to_list(self, data_list: List[OrderedDict]):
        """Wrap single aspectTerms into list."""
        for ann in data_list:
            if "aspectTerms" not in ann.keys():
                continue
            if isinstance(ann["aspectTerms"]["aspectTerm"], list):
                continue
            else:
                ann["aspectTerms"]["aspectTerm"] = [ann["aspectTerms"]["aspectTerm"]]
        return data_list

    def _preprocess(self, raw_data: Tuple[OrderedDict, OrderedDict], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 63_SemEval2014."""
        laptop_doc, restaurant_doc = raw_data
        # xml file is composes only of sentences, 'sentence' contains annotation info
        laptop_data = laptop_doc["sentences"]["sentence"]
        restaurant_data = restaurant_doc["sentences"]["sentence"]

        # cast all terms to list of terms (for cleaner iteration over them later)
        # xml parser outputs list for multiple terms but instance for single term
        laptop_data = self._cast_terms_to_list(laptop_data)
        restaurant_data = self._cast_terms_to_list(restaurant_data)

        texts_data = []
        anns_data = []
        for sent_ann in laptop_data:
            if "aspectTerms" not in sent_ann.keys():
                continue

            text = sent_ann["text"]
            id = sent_ann["@id"]

            text_dict = {"id": str(id) + "LAP", "text": text}
            texts_data.append(text_dict)

            for term in sent_ann["aspectTerms"]["aspectTerm"]:
                term_dict = {"id": str(id) + "LAP", "target": term["@term"], "label": term["@polarity"]}
                anns_data.append(term_dict)

        for sent_ann in restaurant_data:
            if "aspectTerms" not in sent_ann.keys():
                continue

            text = sent_ann["text"]
            id = sent_ann["@id"]

            text_dict = {"id": str(id) + "RES", "text": text}
            texts_data.append(text_dict)

            for term in sent_ann["aspectTerms"]["aspectTerm"]:
                term_dict = {"id": str(id) + "RES", "target": term["@term"], "label": term["@polarity"]}
                anns_data.append(term_dict)

        texts = pd.DataFrame(texts_data)
        annotations = pd.DataFrame(anns_data)
        self._log_before_preprocessing(data=annotations)

        # drop conflict annotations
        annotations = annotations[annotations["label"] != "conflict"]
        df = annotations.merge(texts, on="id", how="left")[["target", "text", "label"]]
        assert all(df.apply(lambda row: row["target"] in row["text"], axis=1))
        df["label"] = df["label"].map(MAPPING).astype(int)

        # Return text, target and label (0, 1, 2)
        df.rename(columns={"target": "pos"}, inplace=True)

        # Remove those observations where the taget is not contained in text (3 after calling ._clean)
        mask = df.apply(lambda row: row["pos"] in row["text"], axis=1)
        df = df[mask]

        df = self._clean(df, length_threshold=length_threshold)
        self._log_after_preprocessing(data=df)
        self.set_logger_data("primary_label_distribution", df["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return df

    def _load_raw_data_from_local(self) -> Tuple[OrderedDict, OrderedDict]:
        """Load the raw data of 63_SemEval2014."""
        # load train, test, dev files

        laptop_path = "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml"
        restaurants_path = "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml"

        with open(os.path.join(self._raw_data_local_path, laptop_path)) as f:
            laptop_doc = xmltodict.parse(f.read())

        with open(os.path.join(self._raw_data_local_path, restaurants_path)) as f:
            restaurant_doc = xmltodict.parse(f.read())

        return laptop_doc, restaurant_doc
