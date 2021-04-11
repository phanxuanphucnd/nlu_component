import re
import logging
import pandas as pd

from unicodedata import normalize as nl
from rasa.shared.nlu.training_data.training_data import TrainingData, Message

logger = logging.getLogger(__name__)

def check_url_exists(txt):

    '''

    Function to check url exists. Return True if exists, otherwise
    :param:
        txt : str: The user's input
    :return:
        True if exists url, otherwise

    '''

    # url_regex = re.compile(r'\bhttps?://\S+\b')
    url_full_regex = re.compile(r'^\W*(?:\s*\bhttps?://\S+\b\W*)+$')

    return re.match(url_full_regex, txt) is not None

def convert_to_denver_format(training_data: TrainingData):
    """Function convert data-format from TrainingData in rasa to denver-format. 
    
    :returns: A DataFrame following denver-format.
    """

    common_examples = []

    logger.debug(f"Convert to Denver format. ")

    training_examples = training_data.training_examples 

    for i in range(len(training_examples)):
        example = {}
        
        example_nlu = training_examples[i].as_dict_nlu()
        example["text"] = example_nlu.get("text", None)
        example["intent"] = example_nlu.get("intent", None)
        example["entities"] = example_nlu.get("entities", None)

        common_examples.append(example)

    final_data = []
    for example in common_examples:
        intent = example['intent']
        
        entities = example.get('entities', None)
        
        text = example['text']
        if text is not None:
            text = nl('NFKC', text).strip()
            text = re.sub(r"\s{2,}", " ", text)

            tags = convert_to_ner(entities, text)

            if len(text.split()) != len(tags.split()):
                logger.warning(f"Length of TEXT = {len(text.split())} different with "
                            f"length of TAGS = {len(tags.split())}, "
                            f"TEXT: {text} - TAGS: {tags}")
            else:
                final_data.append({
                    "text": text, 
                    "intent": intent, 
                    "tags": tags,
                })

    data_df = pd.DataFrame(final_data)
    data_df = pd.DataFrame({
        'text': data_df.text,
        'intent': data_df.intent,
        'tags': data_df.tags
    })
    
    return data_df

def convert_to_ner(entities, text):

    list_text_label = []
    tokens = text.split()

    list_text_label = ['O']*len(tokens)

    if entities == None:
        return ' '.join(list_text_label)

    for info in entities:
        label = info['entity']

        start = info['start']
        end = info['end']

        value = text[start:end]
        list_value = value.split(" ")

        index = len(text[:start].split(" ")) - 1
        list_text_label[index] = 'B-' + str(label)

        for j in range(1, len(list_value)):
            try:
                list_text_label[index + j] = 'I-' + str(label)
            except Exception as e:
                print(str(e))
                print(text)
                print(entities)
                
    return ' '.join(list_text_label)


def cnormalize(text):
    '''Function to normalize text

    :returns: txt: The text after normalize.        
    '''

    # Convert input to UNICODE utf-8
    try:
        txt = nl('NFKC', text)
        # lowercase
        txt = txt.lower().strip()
            
        # Remove emoji
        emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)
        txt = emoji_pattern.sub(r" ", txt) 

        # Remove url, link
        url_regex = re.compile(r'\bhttps?://\S+\b')
        txt = url_regex.sub(r" ", txt)

        # Replace some case
        txt = re.sub(r"[\-]", " - ", txt) 
        txt = re.sub(r"[/]", " / ", txt) 
        txt = re.sub(r"[\n\t\r]", " ", txt) 
        txt = re.sub(r"\s{2,}", " ", txt)

    except Exception as e:
        logger.error(f"  {text}")
        raise ValueError(f"{e}")

    return txt.strip()
