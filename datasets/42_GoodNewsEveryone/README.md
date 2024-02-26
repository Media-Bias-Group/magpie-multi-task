### 42 GoodNewsEveryone
- This dataset contains `5.000` headlines annotated by 5 Crowdworkers per headline.
It contains following columns: `['headline', 'url', 'bias_horizontal', 'bias_vertical', 'country','source', 'dominant_emotion', 'intensity', 'cause', 'experiencer', 'target', 'cue', 'other_emotions', 'reader_emotions']`
The authors point out that with these annotations, one can tackle a total of 6 potential challenges in Emotionality detection:
Emotion classification, Emotion Intensity, Cue or Trigger words, Emotion cause detection, Semantic Role Labeling of Emotions (cue, targeter), Reader vs. Writer vs. Text Perspective
However, they focus on the segmentation tasks, namely sequence labeling of `emotion cues` and `mentions of experiencers`.

We exclude `Emotionality classification` and `Emotionality intensity` because the frequency-distribution are heavily skewed.
We focus on two the segmentation tasks `labeling of experiencers` and `labeling of cues`.
The authors report relatively good F_1 scores for these two tasks and relatively poor performance on the 2 more challenging tasks `Cause detection` and `target detection`.
Both columns for `experiencers` and `cues` contains some very few examples that are badly formatted and can't be properly processed.
We dropped these 3 observations.
We dropped those observations with multiple `cues` and multiple `experiencers`.
We replaced those `cues` and `experiencers` that did not exactly match their counterpart in `headline` with the exact match.
We dropped those where we could not find a match.
- Domain of the columns:
  - `text`: The text.
  - `cue_pos`: The sequence/ cue word that induces the emotion.
  - `experiencer_pos`: The sequence that is the experiencer of the emotion.
- Title: `GodNewsEveryone: A Corpus of News Headlines Annotated with Emotions, Semantic Roles, and Reader Perception`
- Citation Identifier: `bostan_goodnewseveryone_2020`