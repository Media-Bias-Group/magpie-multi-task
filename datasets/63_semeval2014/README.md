### 63 SemEval2014
- aspect-base sentiment analysis (ternary classification)
- original format: two xml files: one with laptop reviews, second with restaurant reviews.
Files are in xml format. Each sentence contains text and then 'aspectTerms' annotations (0-n).
'aspectTerm' annotation is in format 'term' which is a word or small phrase in original sentence and 'polarity' which is a sentiment label for sentence w.r.t. to the term. e.g. "The laptop is nice, but screen is bad". term:laptop, polariy:positive, term:screen, polarity:negative.
These would be two annotations (aspectTerms) for this sentence.
- preprocessing steps:
  - parse xml files
  - extract aspectTerms for each sentence
  - throwaway sentences with no aspecTerms
  - throwaway 'conflict' label - too noisy
  - disregard aspectCategories in restaurants_review file (aspectCategories are abstract kind of aspectTerm. Did not want to mix them so we can train more conveniently)
  - concatenate targets and sentences
- Domains of the labels:
  - `text` : concatenated target (aspectTerm) and sentence (review)
  - `label`: Sentiment label of the `text` w.r.t. target.  `-1=negative, 0=neutral, 1=positive,`
- final size: `5794`
- Citation Identifier: `pontiki_semeval-2014_2014`
- Title: `SemEval-2014 Task 4: Aspect Based Sentiment Analysis`