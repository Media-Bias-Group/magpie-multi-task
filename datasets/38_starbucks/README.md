### 38 Starbucks
- regression
- 5 annotators annotate news sentences such that: 1 = not biased, 2 = slightly biased, 3 = biased, 4 = very biased.
Even though the interagreement score is low regarding 4 classes, overall not biased vs the rest seem reasonable (and their agreement score is not reported).
- preprocessing steps:
  - for each article we have 5 annotators who annotated each sentence, title included. We average those annotations.
  - we clean the text and throw away shorter than 20 sentences
  - we cast the mean of the annotation labels to "continuous scale" via maxmin normalization
- Domain of the labels:
  - `text`: plain text sentence (originally either sentence from the article or the title, mixed together)
  - `label`: regression score on the scale 0-1 (original scale 1 = not biased, 2 = slightly biased, 3 = biased, 4 = very biased)
- final size: `865`
- Citation Identifier: `lim_annotating_2020`
- Title: `Annotating and Analyzing Biased Sentences in News Articles using Crowdsourcing`