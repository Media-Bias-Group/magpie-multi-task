### 108 MeTooMA
- This dataset contains `9973 TweetIDs` collected between October and December 2018 as well as annotations for
`['Country', 'Relevance', 'Directed_Hate', 'Generalized_Hate', 'Sarcasm','Allegation', 'Justification', 'Refutation', 'Support', 'Oppose']`.
These labels fall into 5 categories: `Relevance, stance (Support, Oppose), Hate Speech (directed, generalized), Sarcasm and Dialogue acts (Allegation, Justification, Refutation`.
We were able to retrieve `7874` tweets.
We applied our usual tweet-preparation method, but did not remove Hashtags.
We discarded the columns `Country and Relevance` as they are only useful to filter the dataset for other downstream tasks.
We modelled 4 different subtasks:
  - `Stance detection`: Regression, whether the tweet opposes `(-1)`, supports `(+1)` the MeToo-movement or is neutral `(0)`.
  - `Hate Speech detection`: `Our primary label`. Binary classification. We collapsed the labels for `directed` and `generalized` Hate speech (`1=Hate Speech, 0=no Hate Speech`).
  - `Sarcasm`: Binary classification `(0=no sarcasm, 1=sarcasm)`.
  - `Dialogue acts`: Multi-label classification where each column of `['allegation_label', 'justification_label', 'refutation_label']` contains the binary label indicating its class affiliation.
- Domain of the labels:
  - `text`: The preprocessed text of the tweet.
  - `oppose_support_label`: Multiclass label `(-1=oppose, 0=neutral, 1=support)`.
  - `hate_speech_label`: Binary label.
  - `sarcasm_label`: Binary label.
  - `Remaining columns`: Binary labels indicating class affiliation for multi-label classification.
- Title: `#MeTooMA: Multi-Aspect Annotations of Tweets Related to the MeToo Movement`
- Citation Identifier: `gautam_metooma_2020`