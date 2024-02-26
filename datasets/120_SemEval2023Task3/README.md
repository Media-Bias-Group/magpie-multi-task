### 120 SemEval2023Task3
- This dataset consists of 3 Subtasks.
The first 2 subtasks are article level and hence not of interest to us.
We focus on subtask 3.
The dataset has articles, `paragraphs` and annotations for 6 languages (English, French, German, Italian, Polish, and Russian).
We discarded all languages that are not english.
This subtask has around `9000` english sentences, extracted from news articles.
Each `paragraph` is annotated for the presence of 0, 1 or multiple `persuasion techniques`.
Most persuasion techniques occur very rarely `(eg 8700 negatives, 300 positives)`.
We therefore collapse all persuasion techniques to one `primary label` which indicates the presence of persuasion techniques (or not).
- Domain of the labels:
  - `text` The text of the `paragraph`.
  - `label` Binary label (`1=Presence of persuation techniques, 0=not present`)
  - `All other columns`: Binary labels indicating the presence of a concrete persuasion technique.
- Title: `` TODO
- Citation Identifier: `` TODO