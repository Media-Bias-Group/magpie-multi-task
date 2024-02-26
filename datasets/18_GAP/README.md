### 18 GAP (Gendered Ambiguous Pronouns)
GAP is a high quality, gender-balanced benchmark dataset for coreference resolution of gendered pronouns and contains a total of `4454` multi-sentence sequences.
For each sequence in the collection, it contains one ambiguous `pronoun` as well as two candidate
entities `(A and B)` that occur in the text.
The objective of the `gold-two-mention` task is to classify, which entity `(A or B)` the pronoun refers to.
In some cases (around 10%), the `pronoun` does not refer to `A or B`.
In most cases, `A or B` occur multiple time in the text.
However, the challenge is to align the `pronoun` with exactly the coreferencing occurrence of `A or B`.
We follow `attreeGenderedAmbiguousPronouns2019` and manipulate the raw snippet text by `enclosing the respective span` with a specialized Token.

Example (Most sequences are much longer):
- `original sequence`: Kathleen first appears when Theresa visits her in a prison in London.
- `preprocessed sequence`: <TAG-A> Kathleen <TAG-A> first appears when <TAG-B> Theresa <TAG-B> visits <TAG-P> her <TAG-P> in a prison in London.
- Domain of the labels:
  - `text`: The text with the Tags enclosing spans.
  - `label`: Multiclass classification `(0=Neither, 1=Coreference with entity A, 2=Coreference with entity B)`.
  - `label_gender`: Binary label, not used for classification but to report in-group metric `(0=M, 1=F)`)
- Title: 'Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns'
- Citation Identifier: `webster_mind_2018`
- Citation Identifier for ProBERT: `attreeGenderedAmbiguousPronouns2019` (Winner of Kaggle competition).