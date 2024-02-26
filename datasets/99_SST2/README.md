### 99 SST2
- The Stanford Sentiment dataset is a wellknown collection of approx. `10.000` movie reviews.
It contains fine grained labels describing the sentiment of around `200.000` phrases of the parse trees.
We used the dataset readily downloadable from Huggingface.
This version contains sentence-level binary classification labels only.
We could use that dataset without further preprocessing and are left with `9614` observations.
- Domain of the labels:
  - `text`: The plain text containing a sentence.
  - `label`: The binary label. `0=positive, 1=negative`.
- Citation Identifier: `socher_recursive_2013`
- Title: `Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank`