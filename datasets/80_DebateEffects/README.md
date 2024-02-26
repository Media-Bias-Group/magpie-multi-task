### 80 DebateEffects
- This dataset contains `8000` labeled quote-response pairs derived from debates on 4Forums.com.
Each sentence pair contains an original post as well as a response to that post as well as
annotations along four dimensions: `nasty/ nice, attacking/ reasonable, emotional/ factual and questioning/ asserting`.
Each annotation ranges from -5 to +5, where a higher scores map to the positive end of the scale.

The original dataset contains these columns: `['quote', 'response', 'agree-disagree', 'agreement', 'agreement_unsure',
       'attack', 'attack_unsure', 'defeater-undercutter',
       'defeater-undercutter_unsure', 'fact-feeling', 'fact-feeling_unsure',
       'negotiate-attack', 'negotiate-attack_unsure', 'nicenasty',
       'nicenasty_unsure', 'personal-audience', 'personal-audience_unsure',
       'questioning-asserting', 'questioning-asserting_unsure', 'sarcasm',
       'sarcasm_unsure']`
We extracted only those columns that are related to our emotions-family: `["quote", "response", "fact-feeling", "fact-feeling_unsure"]`
and renamed them to `["text", "response", "label_fact_feeling", "unsure_fact_feeling"]`.
- Domain of the labels:
  - `text`: The raw response (The emotion `label` is w.r.t. this response).
  - `label`: Continuous averaged scale from -5 (emotional) to +5 (factual).
  - `unsure_fact_feeling`: The ratio of the annotators-agreement.
- Citation Identifier: `sridhar_estimating_2019`.
- Title: `Estimating Causal Effects of Tone in Online Debates`.