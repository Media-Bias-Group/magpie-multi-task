### 22_NewsWCL50
- multiclass classification
- Dataset with annotations of media bias from 50 news articles.
Altogether `8696` annotations (on spans) but after processing and aggregating only `736` sentences labeled as LL,L_,M_,R_,RR (political spectrum from far left to far right.)
Eventually remapped to `-2,-1,0,1,2` labels respectively.
Original dataset comprises two files:
Annotations and urls. Urls is the file of the 50 articles with their URLs which you have to fetch by yourself.
Annotation file than includes indices into these articles.
For example paragraph:0, sentence:0 tokens:10-15.
We decided to just take the sentence level therefore we downloaded articles and extracted the sentences with political spectrum labels.
original format:
Annotations.csv: [`id`,`code_type`,`code_name`,`code_mention`,`target_concept`,`event_id`,`publisher_id`,`paragraph`,`sentence`,`start`,`end`]
urls.tsv: [`Event ID`,`Outlet`,`URL`]
- Domain of the labels:
  - `text`: The plain text containing a sentence.
  - `label`: Multiclass label. `-2=far left, -1=left, 0=middle, 1=right,2=far right`.
- Citation Identifier: `hamborg_automated_2019`
- title: `Automated Identification of Media Bias by Word Choice and Labeling in News Articles`