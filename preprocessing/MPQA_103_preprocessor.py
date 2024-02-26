"""This module contains the Preprocessor for the 103 MPQA dataset."""

import os
import re
import zipfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.preprocessors import PreprocessorBlueprint

MAPPING = {"sentiment-pos": 0, "sentiment-neg": 1}


class LeafNode:
    """Structure for loading,holding in memory and processing the MPQA data."""

    def __init__(self, text_path: str, ann_path: str):
        """Initialize the data structures."""
        self.text_path = text_path
        self.ann_path = ann_path
        self._text = None
        self._sentences: List[str] = []
        self._annotations = None

    def _read_text(self):
        """Read plain article text file."""
        with open(self.text_path, "r") as f:
            self._text = f.read()

    def _read_sentences(self):
        """Read sentences (each line one sentence) from file."""
        with open(os.path.join(self.ann_path, "gatesentences.mpqa.2.0"), "r") as f:
            self._sentences = f.readlines()

    def _read_annotations(self):
        """Read annotations (each line one annotation) from file."""
        with open(os.path.join(self.ann_path, "gateman.mpqa.lre.2.0"), "r") as f:
            self._annotations = f.readlines()[5:]  # first 5 lines are always documentation in the file

    def _get_attitude(self, fields: str) -> Optional[int]:
        """Get attitude info from annotation fields. Attitude refers to sentiment."""
        fields = fields.split(" ")
        for field in fields:
            if "attitude-type=" in field:
                label = re.split("attitude-type=", field)[1].strip('"').strip('"\n')
                # only consider sentiment-pos and sentiment-neg annotations
                return MAPPING.get(label)  # return as int 0,1

        return None

    def populate(self):
        """Load all data to data structures."""
        self._read_text()
        self._read_sentences()
        self._read_annotations()

    def get_sentence_spans(self) -> List[Tuple[int, int]]:
        """Extract sentence field as a span."""
        spans = []
        for i, line in enumerate(self._sentences):
            line = line.split("\t")
            left_idx = int(line[1].split(",")[0])
            right_idx = int(line[1].split(",")[1])
            spans.append((left_idx, right_idx))

        return spans

    def get_annotations(self):
        """Extract only span of the annotation with its sentiment label."""
        anns = []
        for line in self._annotations:
            line = line.split("\t")
            left_idx = int(line[1].split(",")[0])
            right_idx = int(line[1].split(",")[1])
            anns.append({"span": (left_idx, right_idx), "sentiment": self._get_attitude(line[4])})
        return anns

    def get_final_sentence(self, span):
        """Extract sentence from article text via its span."""
        return self._text[span[0] : span[1]]


class Preprocessor103MPQA(PreprocessorBlueprint):
    """Preprocessor for the 103 MPQA dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor103MPQA."""
        super(Preprocessor103MPQA, self).__init__(*args, **kwargs)

    def _majority_vote_label(
        self, labels
    ) -> Optional[int]:
        """Get the majority label."""
        return int(np.argmax(np.bincount(labels))) if labels else None

    def _preprocess(self, raw_data: List[LeafNode], length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 103_MPQA."""
        data = []
        final_size = 0

        for node in tqdm(raw_data):
            # for each sentence go through all annotations and find if there is any
            # exclusively inside the sentence. If yes, extract its label. If there
            # are multiple, get majority label
            final_size += len(node.get_annotations())
            for s_span in node.get_sentence_spans():
                annotations = node.get_annotations()
                sentence = node.get_final_sentence(s_span)
                labels = []
                for ann in annotations:
                    # ann span is inside sentence
                    if ann["span"][0] >= s_span[0] and ann["span"][1] <= s_span[1]:
                        if ann["sentiment"] is not None:
                            labels.append(ann["sentiment"])

                final_label = self._majority_vote_label(labels)
                # final label can be undecided (50-50)
                if final_label is not None:
                    data.append({"text": sentence, "label": final_label})

        df = pd.DataFrame(data)
        self._log_before_preprocessing(data=final_size)
        cleaned = self._clean(df, length_threshold=length_threshold)

        self._log_after_preprocessing(data=cleaned)
        self.set_logger_data("primary_label_distribution", cleaned["label"].value_counts().to_dict())
        self.set_logger_data("additional_data", MAPPING)

        return cleaned

    def _load_raw_data_from_local(self) -> List[LeafNode]:
        """Load the raw data of 103_MPQA."""
        with zipfile.ZipFile(os.path.join(self._raw_data_local_path, "mpqa.zip"), "r") as zip_ref:
            zip_ref.extractall(self._raw_data_local_path)

        docs_path = os.path.join(self._raw_data_local_path, "103_mpqa/docs")
        man_anns_path = os.path.join(self._raw_data_local_path, "103_mpqa/man_anns")

        nodes: List[LeafNode] = list()

        # Create LeafNodes which holds the data for each article
        for parent in os.listdir(docs_path):
            for leaf_path in os.listdir(os.path.join(docs_path, parent)):
                # read text,sentence spans and annotations for current leaf
                leaf = LeafNode(
                    os.path.join(docs_path, parent, leaf_path), os.path.join(man_anns_path, parent, leaf_path)
                )
                leaf.populate()
                nodes.append(leaf)

        return nodes

    def _log_before_preprocessing(self, data: int):
        self.set_logger_data("original_size", data)
