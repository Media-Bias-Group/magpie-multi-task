### 101 IMDB
- This dataset contains 25.000 `negative` and `positive` imdb movie reviews.
On imdb, movies can be rated between `0 and 10`.
Movies were assigned a `negative` label, if their review score was `<=4` and a `positive` label if their review score was `>=7`.
We didn't have to do any further processing besides aligning into a pd.DataFrame and storing as a plain .csv-file.
The fina dataset contains evenly balanced movie reviews.
- Domains of the columns:
  - `text`: The text of the review (Sometimes multiple sentences).
  - `label`: Binary label `0=negative, 1=positive`.
- Title: `Learning Word Vectors for Sentiment Analysis`
- Citation Identifier: `maas_learning_2011`