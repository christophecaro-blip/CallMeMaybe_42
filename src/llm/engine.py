"""Single entry point to the language model.

Only this module imports llm_sdk. The rest of the program talks to Engine
with plain types (list[int], np.ndarray), never to the SDK directly.
"""

from pathlib import Path

import numpy as np
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, PrivateAttr

from src.errors import ModelError
from src.llm.vocabulary import Vocabulary


class Engine(BaseModel):
    """Thin wrapper around the SDK model.

    Build it with Engine.load(), which loads the model; the constructor
    alone only stores the vocabulary (useful for tests).

    Attributes:
        vocab: text of every token id, sized to the model's logits.
    """

    vocab: Vocabulary
    # Not a pydantic field: the SDK object is not validated or copied.
    _model: Small_LLM_Model | None = PrivateAttr(default=None)

    @classmethod
    def load(cls) -> "Engine":
        """Load the model and build its vocabulary.

        The logits size is measured once with a first call, so it is
        never hardcoded.

        Returns:
            A ready-to-use engine.

        Raises:
            ModelError: the model or its vocabulary cannot be loaded
                (no network on the first run, disk full, GPU error...).
        """
        try:
            model = Small_LLM_Model()
            ids = model.encode("Hello")[0].tolist()
            size = len(model.get_logits_from_input_ids(ids))
            vocab_path = Path(model.get_path_to_vocab_file())
            vocab = Vocabulary.from_file(vocab_path, size)
        except Exception as e:
            # The SDK can fail in many ways (OSError, RuntimeError from
            # torch, Hugging Face errors...): all become one clear error.
            raise ModelError(f"cannot load the language model: {e}") from e
        engine = cls(vocab=vocab)
        engine._model = model
        return engine

    def _sdk(self) -> Small_LLM_Model:
        """Return the loaded SDK model.

        Raises:
            ModelError: the engine was built without Engine.load().
        """
        if self._model is None:
            raise ModelError("the model is not loaded (use Engine.load())")
        return self._model

    def encode(self, text: str) -> list[int]:
        """Split text into token ids.

        Args:
            text: text to tokenize.

        Returns:
            The token ids, as a flat Python list.

        Raises:
            ModelError: the tokenizer fails.
        """
        try:
            ids: list[int] = self._sdk().encode(text)[0].tolist()
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"cannot encode the text: {e}") from e
        return ids

    def next_logits(self, ids: list[int]) -> np.ndarray:
        """Score every possible next token.

        Args:
            ids: token ids of the text generated so far.

        Returns:
            One float32 score per token id; a higher score means the
            model finds that token more likely.

        Raises:
            ModelError: the model fails (e.g. out of GPU memory).
        """
        try:
            logits = self._sdk().get_logits_from_input_ids(ids)
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"the model failed to run: {e}") from e
        return np.array(logits, dtype=np.float32)
