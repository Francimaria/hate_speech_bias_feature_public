# from https://towardsdatascience.com/build-a-bert-sci-kit-transformer-59d60ddd54a5
from typing import Callable, List, Optional, Tuple

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import torch
from transformers import BertModel, BertTokenizer

class FeatureExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            base_tokenizer,
            base_model,
            max_length: int = 64,
            embedding_func: Optional[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        self.tokenizer = base_tokenizer
        self.model = base_model

        if self.model is None:
            print("Default BERT")
            self.model = BertModel.from_pretrained("bert-base-uncased")
        if self.tokenizer is None:
            print("Default BERT")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :].squeeze()

    def _tokenize(self, text: str) -> Tuple[torch.tensor, torch.tensor]:
        # Tokenize the text with the provided tokenizer
        tokenized_text = self.tokenizer.encode_plus(text,
                                                    add_special_tokens=True,
                                                    max_length=self.max_length,
                                                    truncation=True
                                                    )["input_ids"]

        # Create an attention mask telling the model to use all words
        attention_mask = [1] * len(tokenized_text)

        # model takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str) -> torch.tensor:
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(string) for string in text])

    def fit(self, X, y=None):
        """No fitting necessary so we just return ourselves"""
        return self
