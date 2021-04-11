import os
import wget
import logging
import typing
import shutil
import tempfile

from typing import Any, Optional, Text, Dict
from unicodedata import normalize as nl
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData, Message

from denver.data import DenverDataSource
from denver.embeddings import Embeddings
from denver.learners import FlairSequenceTaggerLearner
from denver.trainers.trainer import ModelTrainer

from salebot_nlu.utils import check_url_exists, convert_to_denver_format

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

logger = logging.getLogger(__name__)

class FlairEntitiesExtractor(Component):
    """A Custom FlairEntitiesExtractor Component. """

    name = "ApolloEntityExtractor"

    provides = ["entities"]
    requires = []
    defaults = {
        "use_pretrain": False,
        "model_repo": "https://tool.dev.ftech.ai/models/apollo_entity_extractor",
        "model_version": "latest",
        "hidden_size": 1024,
        "embedding_type": "bi-pooled_flair_embeddings", 
        "pretrain_embedding": ["vi-forward-1024-lowercase-babe", "vi-backward-1024-lowercase-babe"], 
        "use_crf": True,
        "reproject_embeddings": True, 
        "rnn_layers": 1, 
        "dropout": 0.0,
        "word_dropout": 0.05,
        "locked_dropout": 0.5,
        "batch_size": 32,
        "learning_rate": 0.1, 
        "num_epochs": 500
    }

    language_list = ['vi']

    def __init__(self, component_config=None, learner=None):
        super(FlairEntitiesExtractor, self).__init__(component_config)
        
        self.learner = learner
        self.tempdir = tempfile.mkdtemp()
        self.MODEL_FILE_NAME = f"{__class__.__name__}.pt"

    def __del__(self):
        try:
            # it'll likely fail, but give it a try
            shutil.rmtree(self.tempdir, ignore_errors=True)
        except Exception:
            pass

    def train(self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs):
        if self.component_config["use_pretrain"]:
            logger.debug(f"Use pretrained model for {__class__.__name__}")
        
        else:
            data_df = convert_to_denver_format(training_data)

            data_source = DenverDataSource.from_df(train_df=data_df, 
                                                   text_cols='text', 
                                                   label_cols='tags', 
                                                   lowercase=True)

            embeddings = Embeddings(embedding_types=self.component_config["embedding_type"],
                                    pretrain=self.component_config["pretrain_embedding"])
            embedding = embeddings.embed()

            self.learner = FlairSequenceTaggerLearner(mode='training', 
                                data_source=data_source, 
                                tag_type='ner', 
                                embeddings=embedding, 
                                hidden_size=self.component_config["hidden_size"], 
                                rnn_layers=self.component_config["rnn_layers"], 
                                dropout=self.component_config["dropout"], 
                                word_dropout=self.component_config["word_dropout"], 
                                locked_dropout=self.component_config["locked_dropout"], 
                                reproject_embeddings=self.component_config["reproject_embeddings"], 
                                use_crf=self.component_config["use_crf"])
            
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

                output = self.learner.process(sample=utterance, lowercase=True)
                old_entities = message.data.get("entities")
                for entity in output:
                    old_entities.append(entity)

                message.set("entities", old_entities, add_to_output=True)

    def persist(self, file_name, model_dir):
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again.
        """

        file_name = file_name + ".pt"
        model_file_path = os.path.join(model_dir, file_name)

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
        cached_component: Optional["FlairEntitiesExtractor"] = None,
        **kwargs: Any
    ) -> "FlairEntitiesExtractor":

        """
        Load this component from file.
        """

        file_name = meta.get("file")
        model_file = os.path.join(model_dir, file_name)

        if os.path.exists(model_file):
            learner = FlairSequenceTaggerLearner(mode='inference', model_path=model_file)
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
            model_download_path = model_repo + "/" + model_name + "_" + model_version + ".pt"

            if os.path.exists(model_file):
                os.remove(model_file)

            logger.info("Download file: %s into %s", model_download_path, model_file)
            wget.download(model_download_path, model_file)

        except Exception as e:
            logger.exception("Download model exception: {}".format(e))

        return

    
