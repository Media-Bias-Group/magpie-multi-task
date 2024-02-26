### 105 RtGender
- The authors extracted more than `24852545 top-level responses` from `Facebook, Reddit, Fitocracy and Ted Talks`.
(www.ted.com/talks, www.fitocracy.com).
They also extracted the associated `posts`, but we discarded them for our purposes.
The dataset comes in multiple, large .csv-files.
Each of these sub-datasets is unique as it exhibits specific issues and comes from different distributions.
We refer to the original paper for a detailed explanation.
We discarded the TED-talks-sub-dataset entirely, since we only have the `responses`-file.
We discarded the Reddit-sub-dataset entirely, as the gender was mainly unknown for posts and responses.
We sample `10.000` tuples from `facebook_wiki, facebook_congress and fitocracy` each.
We sample from these `post-response-pairs`, s.t. every post is present at most once in the final dataset.
- Domain of the labels:
  - `post_text`: The text of the post.
  - `text`: The text of the response.
  - `label`: Binary label `(0=M, 1=F)`.
- Title: `RtGender: A Corpus for Studying Differential Responses to Gender`
- Citation Identifier: `voigt_rtgender_2018`