### 104 TRAC-2
- binary classification
- Dataset focused on aggression detection but has also "genderedness" labels.
This NGEN/GEN label means if the sentence is about gender.
We extracted this label as primary target and discarded the aggression label.
Therefore, the classifier will learn to distinguish between sentences talking about gender or not.
- Preprocessing steps:
  - just basic cleaning and keeping only length > 20 sentences
- final size: `3806`
- Domain of the columns:
  - `text`: The plain text containing a sentence.
  - `label`: The binary label. `0=not about gender, 1=about gender`.
- Citation Identifier: `safi_samghabadi_aggression_2020`
- Title: `Aggression and Misogyny Detection using BERT: A Multi-Task Approach`