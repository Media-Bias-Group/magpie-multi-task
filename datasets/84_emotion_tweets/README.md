
### 84 Emotion Tweets
- multiclass classification
- Dataset created from twitter with rule-based algorithm for annotating 8 different emotions. Altogether it consists of 3.5 million tweetIDs and annotations.
- original dataformat: tweetID, label_class
- Preprocessing steps:
  - Sampled 300.000 tweets
  - Fetched them from twitter via twitterAPI.
  - removed duplicates - because a lot of tweets with different IDs (meaning they were for example from different users) were the same
  - basic text cleaning
- Final dataset consists of `197949` datapoints
- Domain of the labels:
  - `text`: The plain text containing a tweet sentence.
  - `label`: The multiclass label: `MAPPING = {
    "Anger": 1,
    "Anticipation": 2,
    "Disgust": 3,
    "Fear": 4,
    "Joy": 5,
    "Sadness": 6,
    "Surprise": 7,
    "Trust": 8,
}`.
- Citation Identifier: `krommyda_experimental_2021`
- Title: `An Experimental Analysis of Data Annotation Methodologies for Emotion Detection in Short Text Posted on Social Media`