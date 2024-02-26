### 25 FakeNewsNet
- binary classification
- The dataset concists originally from urls of fake news articles and real news articles (from 2 different crowdsourcing platforms).
We do not use article level but we decided to extract at least the titles of the news, since its `~23.000` news headlines with fakenews/realnews annotation, which might still be useful.
Original dataformat also included tweets that shared the news. Which we don't think is necessary for us.
- preprocessing steps:
  - extract headlines
  - assign fake news headlines label 1 and real news headlines label 0
- Domain of the labels:
  - `text`: The plain text containing a title of fake news/real news article.
  - `label`: The binary label. `0=real, 1=fake news`.
- final size: `23196`
- Citation Identifier: `shu_fakenewsnet_2020`
- Title: `FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media`