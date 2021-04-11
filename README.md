### Custom Natural Language Understanding (NLU) Component Rasa

#### Setup pipeline

- **Install lib**

```js

pip uninstall denver  # if existed.

pip install denver # from repo: https://github.com/phanxuanphucnd/denver


pip uninstall salebot_nlu  # if existed.

# create lib

python setup.py bdist_wheel

pip install dist/{name of library}

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
