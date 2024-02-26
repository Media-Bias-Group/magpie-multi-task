### 86 OffensiveLanguage
- This dataset contains a total of `24,783` tweets stored in a .csv-file.
3 crowdworkers assigned each tweet into one of the three categories `Hate Speech`, `Offensive Language`, `Neither/ Neutral`.
- Domain of the labels:
  - `text`: The preprocessed tweet.
  - `label`: Multiclass label (`0=Hate Speech, 1=Offensive Language, 0=Neither/Neutral`)
- Title: `Automated Hate Speech Detection and the Problem of Offensive Language`
- Citation Identifier: `davidson_automated_2017`
### 87 OnlineHarassmentDataset
- This dataset contains `20360` tweets that are classified in `hatespeech` or not.
The authors also point out that the dataset might still contain some duplicates though they do not know
how many. However, we apply our usual preprocessing step including `removal of duplicates`.
- Domain of the labels:
  - `text`: The plain tweet.
  - `label`: Binary classification label. `0=neutral, 1=HateSpeech`
- Title: `A Large Labeled Corpus for Online Harassment Research`
- Citation Identifier: `golbeck_large_2017`