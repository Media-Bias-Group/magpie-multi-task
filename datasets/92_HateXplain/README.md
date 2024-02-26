### 92 HateXplain
- This dataset contains `tweets and comments` extracted from Twitter and gab.com respectively.
Each statement was annotated by 3 crowdworkers.
Each crowdworker had to assign a primary label `hatespeech, abusive language or neutral` as well as extract a `sequence` from that statement that `induced the hatespeech or abusive language`.
Furthermore, they assigned one out of 25 `targets`, ie who that tweet or comment is trying to offend.
We discard `919` observations where not at least 2/3 annotators reached agreement regarding the primary label `hatespeech, abusive language or neutral`.
We map all distinct 25 targets to 5 more widely defined categories: `race, religion, gender, economic, minority`.
We extract those words and spans from the original sentence as `rationales` where `all` annotators marked the sequence as relevant to ensure agreement.
Since the raw data did not include full sentences but rather a list of (ordered) tokens (words), we detokenized this sequence.
This yields sentences without any punctuation.
We can perform `3 different tasks`:
1. Multi-class classification between `neutral, offensive and hatespeech`.
2. POS tagging for the inducing sequence(s).
3. Multi-label classification: `race, religion, gender, economic, minority`.
- Domain of the columns:
  - `text`: The text.
  - `rationale_pos`: Those words that induce the hatespeech or offensive language. Empty string if `label==neutral`
  - `label`: Multiclass label (`normal=0`, `offensive=1`, `hatespeech=1`)
  - `label_race, label_religion, label_gender, label_economic, label_minority`: 5 Binary labels (`0=not present, 1=present`)
- Title: `HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection`
- Citation Identifier: `mathew_hatexplain_2021`