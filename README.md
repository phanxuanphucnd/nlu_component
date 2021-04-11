### Custom Natural Language Understanding (NLU) Component Rasa

#### Setup pipeline

- **Install lib**

```js

pip uninstall denver  # if existed.

pip uninstall salebot_nlu  # if existed.

pip install http://minio.dev.ftech.ai/resources-denver-v0.0.2-75854855/denver-0.0.2b0-py3-none-any.whl

pip install http://minio.dev.ftech.ai/denver-salebot-nlu-component-latest-3b3ee6a3/salebot_nlu-1.0.1-py3-none-any.whl

```

**Examples:**

```js

# Configuration for Rasa NLU.
# https://rasa.com/docs/rasa/nlu/components/
language: vi

pipeline:
- name: "WhitespaceTokenizer"
- name: "urlextractor.url_extractor.URLExtractor"
  url_entity_name: "tmp_link"
- name: "salebot_nlu.OneNetNLU"
  num_epochs: 150
- name: "azir.AzirEntitySynonymMapper"
  threshold: 0.85
  general_object_types: 
    - 'coc'
    - 'ghe'
    - 'xe'

```


```js

# Configuration for Rasa NLU.
# https://rasa.com/docs/rasa/nlu/components/
language: vi

pipeline:
- name: "WhitespaceTokenizer"
- name: "urlextractor.url_extractor.URLExtractor"
  url_entity_name: "tmp_link"
- name: "salebot_nlu.FlairEntitiesExtractor"
  use_pretrain: True
- name: "azir.AzirEntitySynonymMapper"
  threshold: 0.85
  general_object_types: 
    - 'coc'
    - 'ghe'
    - 'xe'
- name: "salebot_nlu.ULMFITIntentClassifier"
  use_pretrain: True

```
