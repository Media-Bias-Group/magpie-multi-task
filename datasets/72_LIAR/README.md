### 72 LIAR
- This dataset consists of 3 tsv files (train, valid, test) and contains a total of `12.791` statements scraped from `http://www.politifact.com`.
Each statement contains labels for `subject, context/venue, speaker, state, party, and prior history`.
We do not make use of these variables.
Additionally, each statement is labeled as one of `["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]`.
We map each of the labels to a continuous variable between `0=true and 1=pants-fire`.
We also provide binary labels (`0=true and 1=false`) where we simply round the continuous score to either `0 or 1`.
- Domain of the columns:
  - `text`: The text of the statement.
  - `label`: The continuous score indicating the truthfulness of the statement `0=true and 1=false`.
  - `label_binary`: The binary label `0=true and 1=false`.
- Title: `"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection`
- Citation Identifier: `wang_liar_2017`