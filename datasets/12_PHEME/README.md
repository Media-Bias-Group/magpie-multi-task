### 12 PHEME
- `This dataset was used in the paper 'All-in-one: Multi-task Learning for Rumour Verification'.`
This is not the same as PHEME2017 or PHEME2018.
This dataset contains a total of `5221` Twitter rumours and non-rumours posted during breaking news.
Each rumour is annotated with its veracity, either `True, False or Unverified`.
The dataset comes in a very nested format with `tweetIDs`, `annotations`, `reactions` etc. all in separate files sorted in 9 different `topics`.
Each topic contains directories for `rumour` and `non-rumour` tweets respectively.
Each `rumour-tweet` also contains a file `annotations` with annotations indicating `the truthfulness (veracity)`.
However, not all `rumour-tweets` were annotated correctly.
First, we downloaded the tweets corresponding to the `tweetIDs`.
Then, we assigned each tweet a binary `label` indicating whether it's a `0=non-rumour, 1=rumour`.
Then, for all rumours, we assigned them to one of three classes depending on the truthfulness (`0=false, 1=true, 2=unknown`).
Most likely, we will discard the ambiguous tweets (`varacity_label==2`) and will have a binary classification problem.
- Domain of the labels:
  - `text`: The raw tweet.
  - `label`: Binary label `0=non-rumour, 1=rumour`
  - `veracity_label`: 3 classes `0=false, 1=true, 2=unknown`
- Url where we downloaded the dataset from: `https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078`
- Title: `All-in-one: Multi-task Learning for Rumour Verification`
- Citation Identifier: `kochkina_all--one_2018`