 ### 128_GWSD
-In total, the dataset contains 2010 sentences with labels as neutral, agrees and disagrees wrt to global warming.
There is no prompt or anything, the target of the stance is implicit - global warming.
Originally had 8 annotations per sentence. However, authors also included in the dataset a probability
distribution over the final labels (neutral, agrees, disagrees) inferred as a bayesian probability of the annotators bias etc.
We looked only at this inferred column as a label and took the argmax as a label. However, we discarded those annotations that had
final label below 0.5 (that would happend for example if all the classes had probabilty 0.333..).
- Domain of the labels:
  - `text`: target sentence
  - `label`: Multiclass label indicating the stance `(0=neutral, 1=agree, 2=disagree)`.
- Title: `DeSMOG: Detecting Stance in Media On Global Warming.`
- Citation Identifier: `luoDeSMOGDetectingStance2020`