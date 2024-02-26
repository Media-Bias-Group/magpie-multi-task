### 116 MDGender
- The authors distinguish Gender Bias along 3 dimensions: `about/to/as` for `men` and `women` respectively.
The dataset is very balanced and contains `2345` rows.
Confidence indicates how sure the annotator was that the conversation falls into this category.
Turker gender refers to the gender of the crowdworker. We discard both columns.
Columns: `['text', 'original', 'labels', 'class_type', 'turker_gender', 'confidence'`
`Labels` range from `0-5` and encode all 6 possible combinations of `About/ to/ as x M/W`.
- Domains of the labels:
  - `text`: The text.
  - `label`: Multiclass classification `(0=about w, 1=about m, 2=to w, 3=to m, 4=as w, 5=as m)`.
- Title: `Multi-Dimensional Gender Bias Classification`
- Citation Identifier: `dinan_multi-dimensional_2020`