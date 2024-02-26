### 891 WikiDetoxToxicity
- This dataset contains annotated discussions, extracted from Wikipedia reviews.
It is part of the `Wikipedia Detox Research project` (https://meta.wikimedia.org/wiki/Research:Detox).
This dataset was mentioned in `Stereotypical Bias Removal for Hate Speech Detection Task using Knowledge-based Generalizations. (2017)`.
This dataset is also referred to as `WIKIPEDIA COMMENT CORPUS`.
Each diff was annotated by 10 annotators.
The discussions are distributed in 3 categories/ files, `toxicity` (160k), `aggression` (115k) and `attack` (115k).
Each file contains the following columns: `['rev_id', 'comment', 'year', 'logged_in', 'ns', 'sample', 'split']`.
Furthermore, for each of the 3 categories, there is a file `<category>_annotations.csv` with the original annotations.
For these crowd-worker-level-annotations, we either averaged continuous labels (i.e. in case of `toxicity` and `aggression`) or took the majority vote (i.e. in case of `attack`).
We removed (repeated) occurences of `NEWLINE_TOKEN` from the text.
- We joined the annotated sentences with their respective annotations.
We joined aggression and attack together, since they contained the exact same `observations`.
As we `want and have to` downsample this dataset, we decided to discard exceptionally long observations `(i.e. where length of text > 3rd quartile)`
- Domains of the labels of `toxicity_preprocessed.csv`:
  - `text`: The raw text containing the sentence.
  - `label`: The toxicity score ranging from -2 (highly toxic) to +2.
- Citation Identifier: `wulczyn_ex_2017`
- Title: `Ex Machina: Personal Attacks Seen at Scale`