### 119_SemEval2023Task4
REMEMBER: `Data is provided as tab-separated values files with one header line. We will continuously improve the data until early November 2022.`
- The dataset is part of the SemEval2023 Tasks.
It can be downloaded and inspected from `https://zenodo.org/record/6818093#.Yz1R3NJBxH4`.
As of October 5th, 2022 it contains `5220` observations in the `training` set.
The publishers will update the dataset continuously.
As of today, there are `no test- or dev-sets`.
The authors provide 2 files. One file contains `Premises and Conclusions` as well as a `Stance`, indicating whether the `Premise` agrees with the `Conclusion`.
- Preprocessing: We discarded all the `Value-annotations` and focus only on `Stance-detection`.
- Domain of the labels:
  - `text`: Concatenated Conclusion (target) and premise (sentence with stance wrt target).
  - `label`: Binary label `(1=agree, 0=disagree)`.
- Title: `Touché23-Human-Value-Detection`
- Citation Identifier: `kiesel_johannes_touche23-human-value-detection_2022`