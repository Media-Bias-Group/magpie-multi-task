"""This module contains the Preprocessor for 92_HateXplain dataset."""

import itertools
import json
import os
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd

from preprocessing.preprocessors import PreprocessorBlueprint

DIMENSIONS = {
    "Asexual": "gender",
    "Buddhism": "religion",
    "Hindu": "religion",
    "Heterosexual": "gender",
    "Christian": "religion",
    "Economic": "economic",
    "Indian": "race",
    "Indigenous": "race",
    "Nonreligious": "religion",
    "Minority": "minority",
    "Disability": "minority",
    "Arab": "race",
    "Men": "gender",
    "Hispanic": "race",
    "Islam": "religion",
    "Homosexual": "gender",
    "Jewish": "religion",
    "Women": "gender",
    "Caucasian": "race",
    "Asian": "race",
    "African": "race",
    "Refugee": "minority",
    "Bisexual": "gender",
}
MAPPING = dict(zip(["normal", "offensive", "hatespeech"], range(3)))


class Preprocessor92HateXplain(PreprocessorBlueprint):
    """Preprocessor for the 92_HateXplain dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize a Preprocessor92HateXplain."""
        super(Preprocessor92HateXplain, self).__init__(*args, **kwargs)

    def _preprocess(self, raw_data: Dict, length_threshold: int) -> pd.DataFrame:
        """Preprocess the raw data of 92_HateXplain."""
        self._log_before_preprocessing(data=raw_data)

        new = []
        for k, v in raw_data.items():
            # We have 3 annotators per statement. Each annotator can state multiple targets.
            annotators = v["annotators"]
            rationales = v["rationales"]
            post_tokens = v["post_tokens"]

            labels = [annotator["label"] for annotator in annotators]
            labels_counter = Counter(labels)
            most_common = labels_counter.most_common(1)
            if most_common[0][1] < 2:
                # In that case, we do not have agreement between the annotators and continue
                continue
            label = most_common[0][0]
            label = MAPPING[label]

            targets = [annotator["target"] for annotator in annotators]
            targets = set(
                filter(lambda x: x not in ["None", None, "Other"], list(itertools.chain.from_iterable(targets)))
            )
            targets = list(map(lambda x: DIMENSIONS[x], targets))
            if rationales:
                rationales = np.array(list(map(lambda x: np.array(x), rationales)))
                try:
                    rationales = np.all(rationales, axis=0)
                    diffs = np.diff((rationales).astype(int), axis=0)
                    starts = np.argwhere(diffs == 1)
                    stops = np.argwhere(diffs == -1)
                    starts = list(itertools.chain.from_iterable(starts))
                    stops = list(itertools.chain.from_iterable(stops))
                    rationales = [
                        self._unify_text(text=" ".join((post_tokens)[start:stop]), rm_hashtag=False)
                        for start, stop in list(zip(starts, stops))
                    ]
                except ValueError:
                    continue
                    # In only one case we have rationales that don't have same dimensions.
                    # simply ignore that case

            new.append(
                {
                    "text": " ".join(post_tokens),
                    "rationale_pos": ";".join(rationales),
                    "label": label,
                    "label_gender": 1 if "gender" in targets else 0,
                    "label_religion": 1 if "religion" in targets else 0,
                    "label_race": 1 if "race" in targets else 0,
                    "label_economic": 1 if "economic" in targets else 0,
                    "label_minority": 1 if "minority" in targets else 0,
                }
            )
        df = pd.DataFrame(new)
        df = self._clean(df=df, length_threshold=length_threshold, rm_hashtag=False)
        self._log_after_preprocessing(data=df)
        self.set_logger_data(key="primary_label_distribution", value=df.label.value_counts().to_dict())
        self.set_logger_data(
            key="additional_data",
            value={
                "label_gender_distribution": df.label_gender.value_counts().to_dict(),
                "label_religion_distribution": df.label_religion.value_counts().to_dict(),
                "label_race_distribution": df.label_race.value_counts().to_dict(),
                "label_economic_distribution": df.label_economic.value_counts().to_dict(),
                "label_minority_distribution": df.label_minority.value_counts().to_dict(),
                "MAPPING": MAPPING,
                "DIMENSIONS": DIMENSIONS,
            },
        )
        return df

    def _load_raw_data_from_local(self) -> Dict:
        """Load the raw data of 92_HateXplain."""
        with open(os.path.join(self._raw_data_local_path, "Data", "dataset.json"), "r") as f:
            data = json.load(f)
        return data
