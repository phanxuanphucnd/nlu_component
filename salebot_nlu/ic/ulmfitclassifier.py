import os
import re
import wget
import logging
import typing
import shutil
import tempfile
import copy
import json

from typing import Any, Optional, Text, Dict
from unicodedata import normalize as nl
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData, Message

from denver.data import DenverDataSource
from denver.trainers.language_model_trainer import LanguageModelTrainer
from denver.learners import ULMFITClassificationLearner
from denver.trainers.trainer import ModelTrainer

from salebot_nlu.utils import check_url_exists, convert_to_denver_format

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

logger = logging.getLogger(__name__)

class ULMFITIntentClassifier(Component):
    """A Custom ULMFITIntentClassifier Component. """

    name = "ICIntentClassifier"

    provides = ["intent", "intent_ranking"]
    requires = []

    # Default Metadata
    defaults = {
        'use_pretrain': False,
        'model_repo': 'https://tool.dev.ftech.ai/models/ic_intent_classifier',
        'model_version': 'latest',
        'drop_mult': 0.3,
        'bs_finetune': 128,
        'lr_finetune': 1e-3,
        'num_epochs_finetune': 10,
        'learning_rate': 2e-2,
        'batch_size': 128,
        'num_epochs': 14
    }

    language_list = ['vi']

    def __init__(self, component_config=None, learner=None):
        super(ULMFITIntentClassifier, self).__init__(component_config)

        self.learner = learner
        self.tempdir = tempfile.mkdtemp()
        self.MODEL_FILE_NAME = f"{__class__.__name__}.pkl"

    def __del__(self):
        try:
            # it'll likely fail, but give it a try
            shutil.rmtree(self.tempdir, ignore_errors=True)
        except Exception:
            pass

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs):
        if self.component_config['use_pretrain']:
            logger.debug(f"Use pretrained model for {__class__.__name__}")

        else:
            data_df = convert_to_denver_format(training_data)

            data_source = DenverDataSource.from_df(train_df=data_df, 
                                                   text_cols='text', 
                                                   label_cols='intent', 
                                                   lowercase=True, 
                                                   rm_special_token=True, 
                                                   rm_emoji=True, 
                                                   rm_url=True)

            lm_trainer = LanguageModelTrainer(pretrain='babe')
            lm_trainer.fine_tuning_from_df(data_df=data_source.train.data, 
                                    batch_size=self.component_config["bs_finetune"], 
                                    learning_rate=self.component_config["lr_finetune"], 
                                    num_epochs=self.component_config["num_epochs_finetune"])

            self.learner = ULMFITClassificationLearner(mode='training', 
                                                  data_source=data_source, 
                                                  drop_mult=self.component_config["drop_mult"])
            trainer = ModelTrainer(learn=self.learner)
            trainer.train(base_path=self.tempdir, 
                          model_file=self.MODEL_FILE_NAME, 
                          learning_rate=self.component_config["learning_rate"], 
                          batch_size=self.component_config["batch_size"], 
                          num_epochs=self.component_config["num_epochs"])

    def process(self, message, **kwargs):
        """A method which will parse incoming user messages ."""
        
        if self.learner:
            text = message.data.get('text')
            if text:
                utterance = nl('NFKC', text.strip())
                sample = copy.deepcopy(utterance)
                
                if check_url_exists(utterance):
                    intent = {
                        "name": "inform", 
                        "confidence": 1.0
                    }
                    intent_ranking = [intent]
                
                elif utterance.isdigit() and len(utterance) >= 9:
                    intent = {
                        "name": "handover_to_inbox", 
                        "confidence": 1.0
                    }
                    intent_ranking = [intent]

                elif not sum(c.isalnum() for c in utterance):
                    intent = {
                        "name": "ignore", 
                        "confidence": 1.0
                    }
                    intent_ranking = [intent]
                
                else:
                    output = self.learner.process(sample=sample, 
                                                lowercase=True, 
                                                rm_special_token=True, 
                                                rm_url=True, 
                                                rm_emoji=True)
                    intent = output["intent"]
                    intent_ranking = output["intent_ranking"]

                uncertainty_score = self.learner.get_uncertainty_score(sample=sample, 
                                                                    lowercase=True, 
                                                                    rm_special_token=True, 
                                                                    rm_url=True, 
                                                                    rm_emoji=True)

                intent["uncertainty_score"] = uncertainty_score

                logger.debug(intent)

                message.set("intent", intent, add_to_output=True)
                message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, file_name, model_dir):
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again.
        """

        file_name = file_name + ".pkl"
        model_file_path = os.path.join(model_dir, file_name)

        logger.debug(f"Model file path: {model_file_path}")

        if self.component_config["use_pretrain"]:
            self._fetch_model(model_file_path)
        else:
            temp_path = os.path.join(self.tempdir, self.MODEL_FILE_NAME)
            shutil.move(temp_path, model_file_path)

        return {"file": file_name}

    @classmethod
    def load(
        cls, 
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["ULMFITIntentClassifier"] = None,
        **kwargs: Any
    ) -> "ULMFITIntentClassifier":

        """
        Load this component from file.
        """

        file_name = meta.get("file")
        model_file = os.path.join(model_dir, file_name)

        if os.path.exists(model_file):
            learner = ULMFITClassificationLearner(mode='inference', model_path=model_file)
            return cls(meta, learner)

        else:
            logger.debug(
                f"Failed to load model for tag '{file_name}' for {__class__.__name__}. "
                f"Maybe you did not provide enough training data and no model was "
                f"trained or the path '{os.path.abspath(model_file)}' doesn't "
                f"exist?"
            )

            return cls(meta)


    def _fetch_model(self, model_file):
        try:
            model_name = self.name
            model_version = self.component_config["model_version"]
            model_repo = self.component_config["model_repo"]
            model_download_path = model_repo + "/" + model_name + "_" + model_version + ".pkl"

            if os.path.exists(model_file):
                os.remove(model_file)

            logger.info("Download file: %s into %s", model_download_path, model_file)
            wget.download(model_download_path, model_file)

        except Exception as e:
            logger.exception("Download model exception: {}".format(e))

        return

    