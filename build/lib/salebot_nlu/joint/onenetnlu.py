import logging
import os
import wget
import shutil
import copy
import typing
import tempfile
from typing import Text, Optional, Dict, Any
from unicodedata import normalize as nl

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.shared.nlu.training_data.training_data import TrainingData, Message

from denver.data import DenverDataSource
from denver.learners import OnenetLearner
from denver.trainers.trainer import ModelTrainer

from salebot_nlu.utils import convert_to_denver_format, check_url_exists, cnormalize

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

logging.getLogger('allennlp.training.util').disabled = True
logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.training.trainer').disabled = True
logging.getLogger('allennlp.training.tensorboard_writer').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger('allennlp.common.from_params').disabled = True
logging.getLogger('allennlp.training.checkpointer').disabled = True
logging.getLogger('allennlp.common.checks').disabled = True
logging.getLogger('allennlp.modules.token_embedders.embedding').disabled = True
logging.getLogger('allennlp.data.vocabulary').disabled = True
logging.getLogger('allennlp.training.optimizers').disabled = True
logging.getLogger('allennlp.data.iterators.data_iterator').disabled = True

logger = logging.getLogger(__name__)

class OneNetNLU(IntentClassifier, EntityExtractor):
    name = "OneNetNLU"

    provides = ["intent", "intent_ranking", "entities"]
    requires = []
    defaults = {
        "use_pretrain": False,
        "model_repo": "http://tool.dev.ftech.ai:30001/models/onenet_nlu",
        "model_version": "latest",
        "rnn_type": "lstm", 
        "hidden_size": 200,
        "bidirectional": True, 
        "word_embedding_dim": 50, 
        "word_pretrained_embedding": "vi-glove-50d", 
        "dropout": 0.5, 
        "char_encoder_type": "cnn",
        "char_embedding_dim": 30,
        "num_filters": 128,
        "batch_size": 64, 
        "learning_rate": 0.001, 
        "num_epochs": 150
    }

    def __init__(self, component_config=None, learner=None) -> None:
        super(OneNetNLU, self).__init__(component_config)

        self.learner = learner
        self.tempdir = tempfile.mkdtemp()
        self.MODEL_FILE_NAME = f"{__class__.__name__}.tar.gz"

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Optional[Text] = None,
            model_metadata: Optional["Metadata"] = None,
            cached_component: Optional["Component"] = None,
            **kwargs: Any,
    ) -> "Component":
        """
        load model from file path
        """
        file_name = meta.get("file")

        if not file_name:
            logger.debug(
                f"Failed to load model for {__class__.__name__}. "
                f"Maybe you did not provide enough training data and no model was "
                f"trained or the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)

        model_file = os.path.join(model_dir, file_name)

        if os.path.exists(model_file):
            learner = OnenetLearner(mode="inference", model_path=model_file)
            return cls(meta, learner)
        else:
            logger.debug(
                f"Failed to load model for tag '{file_name}' for {__class__.__name__}. "
                f"Maybe you did not provide enough training data and no model was "
                f"trained or the path '{os.path.abspath(model_file)}' doesn't "
                f"exist?"
            )

            return cls(meta)

    def train(
            self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        if self.component_config["use_pretrain"]:
            logger.debug(f"Use pretrained model for {__class__.__name__}")

        else:
            train_df = convert_to_denver_format(training_data)
            data_source = DenverDataSource.from_df(train_df=train_df,
                                                   text_cols='text',
                                                   intent_cols='intent',
                                                   tag_cols='tags',
                                                   lowercase=True)
            self.learner = OnenetLearner(
                mode='training', 
                data_source=data_source, 
                rnn_type=self.component_config["rnn_type"], 
                dropout=self.component_config["dropout"], 
                bidirectional=self.component_config["bidirectional"], 
                hidden_size=self.component_config["hidden_size"], 
                word_embedding_dim=self.component_config["word_embedding_dim"], 
                word_pretrained_embedding=self.component_config["word_pretrained_embedding"],
                char_embedding_dim=self.component_config["char_embedding_dim"], 
                char_encoder_type=self.component_config["char_encoder_type"], 
                num_filters=self.component_config["num_filters"])

            trainer = ModelTrainer(learn=self.learner)
            trainer.train(base_path=self.tempdir, 
                          model_file=self.MODEL_FILE_NAME, 
                          batch_size=self.component_config["batch_size"], 
                          learning_rate=self.component_config["learning_rate"], 
                          num_epochs=self.component_config["num_epochs"])

    def process(self, message: Message, **kwargs: Any) -> None:
        """
        predict intent & entities from a text message
        :return: rasa nlu output format
        """
        if self.learner:
            text = message.data.get('text')
            output = {}
            if text:
                # if payload == Get started
                if 'get started' in text.lower():
                    intent = {
                        'name': 'start_conversation',
                        'confidence': 1.0
                    }
                    message.set("intent", intent, add_to_output=True)
                    message.set("entities", [], add_to_output=True)
                    return 

                elif check_url_exists(text):
                    intent = {
                        "name": "inform", 
                        "confidence": 1.0
                    }
                    intent_ranking = [intent]
                
                elif text.isdigit() and len(text) >= 9:
                    intent = {
                        "name": "handoff", 
                        "confidence": 1.0
                    }
                    intent_ranking = [intent]

                else:

                    utterance = cnormalize(text)
                    sample = copy.deepcopy(utterance)

                    output = self.learner.process(sample=sample, lowercase=True)

                    intent = output.get("intent")
                    intent_ranking = [intent]
                
                message.set("intent", intent, add_to_output=True)
                message.set("intent_ranking", intent_ranking, add_to_output=True)

                old_entities = message.data.get("entities")
                entities = output.get("entities", [])
                for entity in entities:
                    entity.update({'extractor': 'OneNetNLU'})
                    old_entities.append(entity)

                message.set("entities", old_entities, add_to_output=True)

                message.set("text", output.get('text', text))

        else:
            logger.debug(f'Have no {__class__.__name__} to process message {message.data.get("text")}')

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """
        Persist this model into the passed directory.
        Returns the metadata necessary to load the model again.

        customize: move trained model from temp_dir to persisting location
        """
        file_name = file_name + ".tar.gz"
        model_file_path = os.path.join(model_dir, file_name)

        if self.component_config["use_pretrain"]:
            self._fetch_model(model_file_path)
        else:
            temp_path = os.path.join(self.tempdir, self.MODEL_FILE_NAME)
            shutil.move(temp_path, model_file_path)

        return {"file": file_name}

    def _fetch_model(self, model_file):
        try:
            model_name = self.name
            model_version = self.component_config["model_version"]
            model_repo = self.component_config["model_repo"]
            model_download_path = model_repo + "/" + model_name + "_" + model_version + ".tar.gz"

            if os.path.exists(model_file):
                os.remove(model_file)

            logger.info("Download file: %s into %s", model_download_path, model_file)
            wget.download(model_download_path, model_file)

        except Exception as e:
            logger.exception("Download model exception: {}".format(e))

        return
