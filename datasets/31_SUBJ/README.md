### 31_SUBJ
- binary classification
- A dataset for explicit subjectivity detection. Very simple one. Original format: two files one with 5k objective sentences and second with 5k subjective
- pre-processing steps: putting it into DataFrame and shuffle with seed=42
- Final dataset consists of `10000` sentences, 5k each label
- Domains of the columns:
  - `text`: The plan text (sentence).
  - `label_bias`: binary label 0=objective,1=subjective.
- Citation Identifier: `pang_sentimental_2004`
- Title: `A sentimental education: sentiment analysis using subjectivity summarization based on minimum cuts`