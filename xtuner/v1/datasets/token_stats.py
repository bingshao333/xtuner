import copy
import hashlib
import sys
from dataclasses import dataclass
from typing import Any

from .config import BaseTokenizeFnConfig
from .data_item import CacheItem, DataItem
from .ftdp import MAX_LEN, FtdpTokenizeFunction
from .pt_tokenize_fn.text import PretrainTokenizeFunction
from .sft_tokenize_fn.openai import OpenaiTokenizeFunction
from .utils import CachableTokenizeFunction, with_proxy_attention_flops


class TokenStatsTokenizeFunction(CachableTokenizeFunction[DataItem]):
    """Record token lengths before truncation.

    Args:
        tokenize_fn (CachableTokenizeFunction): OpenAI, FTDP or standard pretraining tokenization function.
    """

    record_token_stats = True

    def __init__(self, tokenize_fn: CachableTokenizeFunction) -> None:
        # Subclasses may truncate before returning token IDs.
        if type(tokenize_fn) not in (OpenaiTokenizeFunction, FtdpTokenizeFunction, PretrainTokenizeFunction):
            raise TypeError(f"Token statistics do not support {type(tokenize_fn).__name__}")
        super().__init__(
            tokenize_fn.tokenizer,
            llm_pack_weight=tokenize_fn.llm_pack_weight,
            visual_pack_weight=tokenize_fn.visual_pack_weight,
        )
        self._source = tokenize_fn
        self._full = copy.copy(tokenize_fn)
        self.max_length: int | None = None
        if type(tokenize_fn) is OpenaiTokenizeFunction:
            self.max_length = tokenize_fn.max_length
            self._full.max_length = None
        elif type(tokenize_fn) is FtdpTokenizeFunction:
            self.max_length = tokenize_fn.max_length or tokenize_fn.template_config.get("max_len", MAX_LEN)
            # FTDP falls back to the template limit when max_length is None or 0.
            self._full.max_length = sys.maxsize
        if self.max_length is not None and self.max_length < 1:
            raise ValueError("Token statistics require a positive effective max_length")
        # Calculate proxy flops after truncation.
        self._full.set_state("runtime")
        self._call_full = type(tokenize_fn).__call__
        self._process_labels = getattr(tokenize_fn, "process_labels", None)
        if self._process_labels is not None:
            # Apply the empty-think label mask after truncation.
            self._call_full = self._call_full.__wrapped__

    @with_proxy_attention_flops
    def __call__(self, item: dict | list, **kwargs: Any) -> DataItem | CacheItem:
        data = self._call_full(self._full, item, **kwargs)
        if "input_ids" not in data:
            # Damaged samples have no measured original length.
            return data
        original = len(data["input_ids"])
        if self.max_length is not None and original > self.max_length:
            data["input_ids"] = data["input_ids"][: self.max_length]
            data["labels"] = data["labels"][: self.max_length]
        data["num_tokens"] = len(data["input_ids"])
        if self._process_labels is not None:
            data["labels"] = self._process_labels(data["input_ids"], data["labels"])
        data["original_num_tokens"] = original
        return data

    def hash(self) -> str:
        """Return the statistics cache key.

        Returns:
            str: Cache key including the source key, effective limit and recorder version.
        """
        key = f"{self._source.hash()}:{self.max_length}:token_stats_v1"
        return "token_stats_v1_" + hashlib.sha256(key.encode()).hexdigest()[:32]


@dataclass
class TokenStatsConfig:
    """Wrap a tokenization config to record original lengths.

    Args:
        tokenize_fn (BaseTokenizeFnConfig): The existing training tokenization config.
    """

    tokenize_fn: BaseTokenizeFnConfig

    def build(
        self, tokenizer: Any, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs: Any
    ) -> TokenStatsTokenizeFunction:
        """Build the recorder using the existing config.

        Args:
            tokenizer (Any): Existing tokenizer.
            tokenizer_hash (str | None): Optional existing tokenizer hash.
            anno_name (str | None): Annotation name forwarded to the config.
            **kwargs (Any): Additional config arguments.

        Returns:
            TokenStatsTokenizeFunction: Opt-in tokenizer with statistics metadata.
        """
        return TokenStatsTokenizeFunction(
            self.tokenize_fn.build(tokenizer, tokenizer_hash=tokenizer_hash, anno_name=anno_name, **kwargs)
        )
