### 96 BU-NEmo
- multiclass classification
- Dataset of emotional responses to headlines, headlines with images or just images from news. They show a headline/headline with image to annotator and they then self-annotate their response with emotion they feel, intenisty of the emotion, feeling and a textual response eg. 'this is just sad'. The data also included political inclination of the annotator and their time for annotation. The comment is what we classify. Original data are split to 'text_only', 'image_only', 'text_image', with regard to what was shown to annotators. Even though the 'text_only' and 'text_image' overlap in headlines, the responses are distinct. We kept only the emotion label since the feeling label was too diverse and other, even though informative, ilabels are not important for this type of task.
- Final dataset consists of 15609 datapoints
- Domain of the labels:
  - `text`: The plain text containing a sentence.
  - `label`: The multiclass label: `MAPPING = {
    "Amusement": 0,
    "Anger": 1,
    "Awe": 2,
    "Contentment": 3,
    "Disgust": 4,
    "Excitement": 5,
    "Fear": 6,
    "Sadness": 7,
}`.
- Citation Identifier:`reardon_bu-nemo_2022`
- Title: `BU-NEmo: an Affective Dataset of Gun Violence News`