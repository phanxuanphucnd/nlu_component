import os
from typing import Any, List, Optional, Dict, Text

import rasa.shared.utils.io
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor
from rasa.nlu.model import Metadata
from rasa.shared.nlu.constants import ENTITIES
from rasa.shared.nlu.training_data.message import Message


class XegexEntityExtractor(RegexEntityExtractor):
    """
    custom rasa regex entity extractor
    we modify only `process` function
    """

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        patterns: Optional[List[Dict[Text, Text]]] = None,
    ):
        super(XegexEntityExtractor, self).__init__(component_config)

        self.case_sensitive = self.component_config["case_sensitive"]
        self.patterns = patterns or []

    def process(self, message: Message, **kwargs: Any) -> None:
        """
        update list entities extracted from regex:
         - add confidence score by 1.0

        merge the entities extracted from regex and other extractor,
        if the same pattern (a substring in message.text) is extracted by many extractor,
        we prefer using output from Regex component
        """
        if not self.patterns:
            return

        extracted_entities = self._extract_entities(message)
        extracted_entities = self.add_extractor_name(extracted_entities)

        new_entities: list = message.get(ENTITIES, [])  # filled with old entities

        for regex_entity in extracted_entities:
            regex_entity['confidence'] = 1.0  # update confidence score

            if not message.get(ENTITIES, []):
                new_entities.extend(extracted_entities)

            is_valid = True
            for old_entity in message.get(ENTITIES, []):  # this iteration made decision to use regex output or not
                if old_entity['extractor'] == 'URLExtractor' or old_entity.get('entity') == 'tmp_link':
                    continue
                elif is_duplicated(regex_entity, old_entity):
                    # prefer regex output
                    new_entities.remove(old_entity)
                elif is_overlap(regex_entity, old_entity):
                    # add regex output
                    is_valid = False

            if is_valid:
                new_entities.append(regex_entity)

        message.set(
            ENTITIES, new_entities, add_to_output=True
        )

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Optional[Text] = None,
            model_metadata: Optional[Metadata] = None,
            cached_component: Optional["XegexEntityExtractor"] = None,
            **kwargs: Any,
    ) -> "XegexEntityExtractor":

        file_name = meta.get("file")
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            patterns = rasa.shared.utils.io.read_json_file(regex_file)
            return XegexEntityExtractor(meta, patterns=patterns)

        return XegexEntityExtractor(meta)


def is_duplicated(e1: Dict, e2: Dict):
    """
    check if 2 entities are shared the same index
    """
    return e1['start'] == e2['start'] and e1['end'] == e2['end']


def is_overlap(e1: Dict, e2: Dict):
    """
    check if 2 entities are overlapping in text index
    """
    return e1['start'] <= e2['start'] <= e1['end'] or e2['start'] <= e1['start'] <= e2['end']
