### 19_MultiDimNews:
- A collection of 90 articles (`2057` sentences) stored in a json-format with detailed annotations for `bias`, `hidden assumptions`, `framing` and `subjectivity`.
Each article has the following keys: `['id', 'sentences', 'subjectivity', 'hidden_assumptions', 'framing', 'bias']`.
Since we are not interested in article-level annotations, we discarded these annotations and instead used the collapsed binary labels already provided by the authors.
- Domains of the columns:
  - `text`: The plain text.
  - `label_bias`: Binary label, whether the text is biased.
  - `label_subj`: Binary label, whether the text is subjective.
  - `label_framing`: Binary label, whether the text contains framing.
  - `label_hidden_assumpt`: Binary label, whether the text contains hidden assumptions.
- Citation Identifier: `farber_multidimensional_2020`