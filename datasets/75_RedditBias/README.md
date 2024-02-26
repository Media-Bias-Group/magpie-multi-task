### 75 RedditBias
- This dataset contains annotated `comments` in 5 csv files (`orientation, gender (female), religion (jewish), religion (muslim), race (black)`).
Each `comment` has a label indicating whether it is `biased` or not.
Additionally, for each `comment`, the authors extracted a `phrase` that - if seen alone - is also labeled as `biased` or `not`.
We decided to split the dataset between the three families `Racial Bias`, `Gender Bias` and `Group Bias`.
- Preprocessing: We have 4 cases:
1. Comment biased, phrase non-biased: We discarded those `comments` that were labeled as `biased` while the extracted phrase was labeled as `non biased`.
For these observations, we could not find a way to integrate them in our `final dataset` without introducing noise.
Therefore we dropped: orientation (18), gender_female (19), religion_jew (788), religion_muslim (37), race_black (42)
2. Comment biased, phrase biased: Straightforward.
3. Comment non-biased, phrase non-biased: We excluded the phrase.
4. Comment non-biased, phrase biased: We included these observations as they incorporate lot's os information.
- Domain of the labels:
  - `text`: The raw, full comment.
  - `bias_pos`: The POS that contains the bias inducing tokens.
  - `label`: Binary label, whether the text is biased (`0=non-biased, 1=biased`)
  - `label_group`: Multiclass label, which group this text/ phrase is potentially targeting `(0=orientation, 1=gender, 2=religion_jew, 3=religion_muslim, 4=race)`
- Title: `RedditBias: A Real-World Resource for Bias Evaluation and Debiasing of Conversational Language Models`
- Citation Identifier: `barikeri_redditbias_2021`