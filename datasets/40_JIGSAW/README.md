### 40 JIGSAW
- This dataset contains `1.999.516` observations and comes in a .csv file with these columns: `['id', 'comment_text', 'split', 'created_date', 'publication_id',
       'parent_id', 'article_id', 'rating', 'funny', 'wow', 'sad', 'likes',
       'disagree', 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit',
       'identity_attack', 'insult', 'threat', 'male', 'female', 'transgender',
       'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
       'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu',
       'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian',
       'latino', 'other_race_or_ethnicity', 'physical_disability',
       'intellectual_or_learning_disability', 'psychiatric_or_mental_illness',
       'other_disability', 'identity_annotator_count',
       'toxicity_annotator_count'],
      dtype='object')
`
We extract the columns `comment_text` and `toxicity` as well as the `identity-mention`-columns.
Furthermore, we extract the column `toxicity_annotator_count` in order to include/ exclude rows in a later step.
We are not interested in the `reactions` on that comment and can not come up with an idea how to use these reactions.
Therefore, we exclude `reactions` entirely.
Label values range from 0 to 1, representing the fraction of the raters that saw this identity being mentioned in the text.
We exclude those observations where no annotations for identity are present.
After that, we are left with `448.000` observations.
The distribution of `toxicity_annotator_count` is heavily skewed with a majority of the comments being labeled by 4 or less annotators.
To further reduce the amount of datapoints and ensure robust labels,
we include datapoints in our final collection if their `toxicity_annotator_count` exceeded the 3rd Quartile, ie 9 or more annotations are present.
After that, we are left with `119.018` rows.
We aggregate the `identity-mention-columns` along 4 dimensions: `gender`, `religion`, `ethnicity` and `disability`.
The value of each of these 4 labels take the value 1 if at least one of the respective columns contains a value geq 0,5.
- Domain of the labels:
  - `text`: The plain text.
  - `label`: Binary label: `0 (non-toxic), 1 (toxic)`.
  - `gender_label`: Binary label: `0 (no mention of identity gender), 1 (mention)`.
  - `religtion_label`: Binary label: `0 (no mention of identity religion), 1 (mention)`.
  - `ethnicity_label`: Binary label: `0 (no mention of identity ethnicity), 1 (mention)`.
  - `disability_label`: Binary label: `0 (no mention of identity disability), 1 (mention)`.
We can use the `label` for classification.
We can use the 4 `identity-labels` for multiclass-classification.
- Citation Identifier: `jigsawconversation_ai_jigsaw_2019`
- Title: `Jigsaw unintended bias in toxicity classification.`