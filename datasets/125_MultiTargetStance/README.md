### 125_MultiTargetStance
- This dataset consists of multitarget label stance classification data. Each text has two targets and two stance labels wrt to each respective target. Eg: "dont vote for him" could have two targets: Trump with stance "against" and Hillary with stance "Favor". Since the targets are not exact substrings of the text, we decided to extract such sentences as two datapoints. Each with respective target and target label only.
- preprocessing steps:
  - Fetching the tweets
  - splitting the datapoints as described above
  - cleaning, preserving hashtags
- final size: 4536
- Domain of the labels:
  - `text` : concatenated target and sentence that represents the stance to the target
  - `label` : label of the stance `{"AGAINST": -1, "NONE": 0, "FAVOR": 1}`
- Title: `sobhaniDatasetMultiTargetStance2017`
- Citation Iden