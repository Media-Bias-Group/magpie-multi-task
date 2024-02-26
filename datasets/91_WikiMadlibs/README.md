### 91 WikiMadlibs
- This dataset contains `76.564` sentences annotated as toxic or non-toxic.
Every sentence is constructed from templates using different `names`, `adjectives` and `verbs`.
The dataset comes in a .csv-file with these columns: `['template', 'toxicity', 'phrase']`, where `template` indicates the structure of the sentence, e.g. `verb_adj`.
We excluded those sentences that consisted of a `verb` and `adjective` only, as they by definition aren't proper sentences.
As all sentences didn't contain any punctuation, we manually appended a dot (`.`) to the end of every sentence.
- Domain of the labels:
  - `text`: The plain text.
  - `label`: The binary label. `0=non-toxic, 1=toxic`
- Citation Identifier: `dixon_measuring_2018`
- Title: `Measuring and Mitigating Unintended Bias in Text Classification`