### 126_WTWT
- Yet another dataset for `multi-target stance detection`.
In total, the dataset contains `51.284` `tweet IDs` as well as a `target` and `stance`.
`Stance` can be either of `comment, unrelated, refute, support`.
We were able to retrieve `42.919` of the original tweets.
Initial frequencies: `comments (17.644), unrelated (15.821), support (5756), refute (3695)`.
We discarded all `unrelated` tweets and treat `comments` as `neutral` stance.
The targets are `5 company-company pairs (Buyer, Target)` for large M&As and not explicitly given in natural language.
To meaningfully encode these targets, we replace them with the full company name.
We removed all duplicates.
- Domain of the labels:
  - `text`: The preprocessed text of the tweet, prepended with the target-prompt.
  - `label`: Multiclass label indicating the stance `(0=refute, 1=neutral, 2=support)`.
- Title: `Will-They-Won't-They: A Very Large Dataset for Stance Detection on Twitter.`
- Citation Identifier: `conforti_will-they-wont-they_2020`