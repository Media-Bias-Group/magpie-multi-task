### 9_BASIL:
- A collection of `300` articles sampled between 2010 and 2019.
For each article, the authors provide 2 files: An article file and a file containing annotations.
Each article object contains the following keys: `['title', 'keywords', 'date', 'uuid', 'url', 'main-entities', 'word-count', 'source', 'main-event', 'triplet-uuid', 'body-paragraphs']` where the 'body-paragraph' contains the sentences of each paragraph.
Each sentence-level-annotation objet contains the following keys: `['aim', 'bias', 'end', 'id', 'indirect-ally-opponent-sentiment', 'indirect-target-name', 'notes', 'polarity', 'quote', 'speaker', 'start', 'target', 'txt']`.
For our purposes, we extracted the labels `bias`, `aim`, `quote`, `txt` from each annotation.
Additionally, we extracted the sentences contained in body-paragraphs from each article.
We entirely discarded the article-level annotations.
With these labels, we perform binary classification for two targets `(bias and aim)`.
Additionally, we can perform POS tagging for the bias-inducing POS `(txt)`.
- Domains of the columns:
  - `label`:  Multiclass label, type of bias `(1=informational, 0=lexical, 2=non-biased)`
  - `aim`: Whether a phrase is directly/ indirectly aiming at the target. `(1=dir, 0=not-dir,2=not-biased)`
  - `pos`: The sequence that induces the bias. This sequence is part of the sentence.
  - `text`: The source sentence.
- Note: The authors used the term `phrase` instead of `sentence`.
To ensure readability and comparability to other datasets, we used the term `text`.
- Citation Identifier: `fan_plain_2019`
- Title: `In Plain Sight: Media Bias Through the Lens of Factual Reporting.`