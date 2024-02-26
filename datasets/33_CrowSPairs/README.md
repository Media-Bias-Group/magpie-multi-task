### 33 CrowSPairs
`Crowedsourced Stereotype Pairs`.
- It comes in a .csv-file containing these columns: `['sent_more', 'sent_less', 'stereo_antistereo', 'bias_type', 'annotations', 'anon_writer', 'anon_annotators']`.
It contains `1.508` examples of sentence pairs where `sent_less` refers to a historically disadvantaged group and can either violate or demonstrate a stereotype.
The second sentence minimally diverges from the first by exchanging one word to identify the group.
`stereo_antistereo` indicates whether the `sent_more` demonstrates or violates the stereotype.
`bias_type` can take either of these values
`['race-color', 'socioeconomic', 'gender', 'disability',
       'nationality', 'sexual-orientation', 'physical-appearance',
       'religion', 'age']`.
It indicates which type of bias is present in the example and is the result of the majority vote of the crowdworkers (see the other 3 columns.)
Again we split the dataset along `sent_more` and `sent_less` and create a binary variable for the presence of a stereotype (`sent_more=stereotype, sent_less=no stereotype`).
We also use the `bias_type` variables for multiclass classification.
We discarded `stereo_antistereo` entirely.
Additionally, we also computed the diffs to predict which tokens in the sentence induce the stereotypes.
- Domains of the columns:
  - `text`: The text containing the sentence with or without stereotype.
  - `label`: Binary label (`0=no stereotype, 1=stereotype`)
  - `stereotype_label`: Multiclass labels (`{'race-color': 0, 'socioeconomic': 1, 'gender': 2, 'disability': 3, 'nationality': 4, 'sexual-orientation': 5, 'physical-appearance': 6, 'religion': 7, 'age': 8}`)
  - `pos`: The words that induce the stereotypes. These words are part of the sentence.
- Title: `CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models`
- Citation identifier: `nangia_crows-pairs_2020`