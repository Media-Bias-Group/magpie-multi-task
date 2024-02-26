### 26_Neutralizing WNC
- A collection of sentence pairs, extracted from Wikipedia Revisions pre- and post-neutralization.
The authors extracted a total of `423.823` revisions between 2004 and 2019.
After preprocessing and filtering, their dataset contains `180.000` biased and neutralized sentence pairs as well as over `385.000` neutral examples of adjacent sentences that were not revised.
They follow Recasens, Danescu-Niculescu-Mizil, and Jurafsky (2013), and only include sentences, where a `single word` from the source text was edited.
Their final collection contains `53.803` observations which contain the following labels:
`["id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "src_POS_tags", "tgt_parse_tags"]`.
`Their train and development datasets were sampled from those training examples`.
We decided to use the raw text as the tokenized text often contained non-interpretable symbols.
In our preprocessing step, we extracted those words from the neutral target sentence that were not contained in the source sentence as our `bias_inducing_words`.
- Domain of the labels:
  - `text_biased`: The raw text containing the biased sentence.
  - `text_neutral`: The raw text containing the debiased sentence.
  - `pos`: A list of words that were removed or replaced in the debiased sentence.
- Citation Identifier: `pryzant_automatically_2020`
- Title: `Automatically Neutralizing Subjective Bias in Text.`