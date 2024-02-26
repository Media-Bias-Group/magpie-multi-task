### 109 Stereotype
- This dataset contains `2221` sentences annotated for the presence of `explicit stereotype` or `implicit stereotypical association`.
The authors define implicit stereotypical association as `Text that is not mainly intended to convey a stereotype but nevertheless propagates a stereotypical association.`
The authors sampled candidate expressions from the subreddits `/r/Jokes` and `/r/AskHistorians` and label the sentences using Amazon Turk.
We collapse `explicit and implicit` stereotypes.
- Domain of the labels:
  - `text`: The plain text containing a statement.
  - `label`: The binary label. `0=no stereotype, 1=stereotype`
  - `stereotype_explicit_label`: The binary label. `0=no explicit stereotype, 1=explicit stereotype`
  - `stereotype_implicit_label`: The binary label. `0=no implicit stereotype, 1=implicit stereotype`
- Citation Identifier: `pujari_reinforcement_2022`
- Title: `Reinforcement Guided Multi-Task Learning Framework for Low-Resource Stereotype Detection`