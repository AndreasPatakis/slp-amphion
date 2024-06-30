# Copyright (c) 2023 Amphion.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List, Tuple
import os
import numpy as np
import torch
from text.symbol_table import SymbolTable
from text import text_to_sequence
import json



"""
    TextToken: map text to id
"""


# TextTokenCollator is modified from
# https://github.com/lifeiteng/vall-e/blob/9c69096d603ce13174fb5cb025f185e2e9b36ac7/valle/data/collation.py
class TextTokenCollator:
    def __init__(
        self,
        text_tokens: List[str],
        add_eos: bool = True,
        add_bos: bool = True,
        pad_symbol: str = "<pad>",
        bos_symbol: str = "<bos>",
        eos_symbol: str = "<eos>",
    ):
        original_symbol_to_id = {
            '<pad>': 0, '<bos>': 1, '<eos>': 2, '!': 3, '"': 4, '(': 5, ')': 6, ',': 7, '.': 8, '1': 9, ':': 10, ';': 11,
            '<eps>': 12, '?': 13, '_': 14, 'a': 15, 'aɪ': 16, 'aɪw': 17, 'aɪə': 18, 'aɪɚ': 19, 'aɪʊ': 20, 'aɪʊɹ': 21, 'aʊ': 22,
            'aʊə': 23, 'aː': 24, 'b': 25, 'd': 26, 'dh': 27, 'dt': 28, 'dʑ': 29, 'dʒ': 30, 'e': 31, 'enus': 32, 'es': 33, 'eə': 34,
            'eɪ': 35, 'f': 36, 'fr': 37, 'fɛ': 38, 'h': 39, 'hi': 40, 'hy': 41, 'i': 42, 'iə': 43, 'iː': 44, 'iːd': 45, 'iːː': 46, 'j': 47,
            'k': 48, 'kh': 49, 'ko': 50, 'l': 51, 'm': 52, 'n': 53, 'nʲ': 54, 'o': 55, 'oʊ': 56, 'oː': 57, 'oːɹ': 58, 'oːː': 59, 'p': 60, 'ph': 61,
            'q': 62, 'r': 63, 's': 64, 't': 65, 'ta': 66, 'tw': 67, 'tɕ': 68, 'tʃ': 69, 'u': 70, 'uː': 71, 'uːj': 72, 'v': 73, 'w': 74, 'x': 75,
            'z': 76, '¡': 77, '«': 78, '»': 79, '¿': 80, 'æ': 81, 'æː': 82, 'ç': 83, 'ð': 84, 'ø': 85, 'ŋ': 86, 'ɐ': 87, 'ɐː': 88, 'ɑ': 89,
            'ɑː': 90, 'ɑːɹ': 91, 'ɒ': 92, 'ɔ': 93, 'ɔɪ': 94, 'ɔː': 95, 'ɔːɹ': 96, 'ɔːɹt': 97, 'ɖ': 98, 'ə': 99, 'əl': 100, 'ən': 101, 'əʊ': 102,
            'ɚ': 103, 'ɛ': 104, 'ɛɹ': 105, 'ɛː': 106, 'ɜː': 107, 'ɡ': 108, 'ɡʲ': 109, 'ɣ': 110, 'ɪ': 111, 'ɪɹ': 112, 'ɪː': 113, 'ɫ': 114, 'ɬ': 115,
            'ɭ': 116, 'ɯ': 117, 'ɲ': 118, 'ɹ': 119, 'ɾ': 120, 'ʁ': 121, 'ʃ': 122, 'ʃm': 123, 'ʈ': 124, 'ʉ': 125, 'ʊ': 126, 'ʊə': 127, 'ʊɹ': 128,
            'ʌ': 129, 'ʒ': 130, 'ʔ': 131, 'ʰχ': 132, 'ʲ': 133, '̃': 134, '̩': 135, 'β': 136, 'θ': 137, 'ᵻ': 138, '—': 139, '…': 140
            }
            
        mapping = {"u":"u:", "c":"k", "ʎ":"l", "oɪ":"ɔɪ", "ʂ":"s", "ts":"tʃ"}
        
        self.pad_symbol = pad_symbol
        self.add_eos = add_eos
        self.add_bos = add_bos
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        unique_tokens = [pad_symbol]
        if add_bos:
            unique_tokens.append(bos_symbol)
        if add_eos:
            unique_tokens.append(eos_symbol)
        unique_tokens.extend(sorted(text_tokens))

        # OUR CODE

        new_phoneme_to_id = {}
        # Assign IDs to the new phonemes
        for phoneme in unique_tokens:
            if phoneme in original_symbol_to_id:
                new_phoneme_to_id[phoneme] = original_symbol_to_id[phoneme]
            else:
                new_phoneme_to_id[phoneme] = original_symbol_to_id[mapping[phoneme]]
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(new_phoneme_to_id)
        #  END OF OUR CODE

        self.token2idx = {phoneme: idx for phoneme, idx in new_phoneme_to_id.items()}
        self.idx2token = {idx: phoneme for phoneme, idx in new_phoneme_to_id.items()}

    def index(self, tokens_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs, seq_lens = [], []
        for tokens in tokens_list:
            assert all([True if s in self.token2idx else False for s in tokens]) is True
            seq = (
                ([self.bos_symbol] if self.add_bos else [])
                + list(tokens)
                + ([self.eos_symbol] if self.add_eos else [])
            )
            seqs.append(seq)
            seq_lens.append(len(seq))

        max_len = max(seq_lens)
        for k, (seq, seq_len) in enumerate(zip(seqs, seq_lens)):
            seq.extend([self.pad_symbol] * (max_len - seq_len))

        tokens = torch.from_numpy(
            np.array(
                [[self.token2idx[token] for token in seq] for seq in seqs],
                dtype=np.int64,
            )
        )
        tokens_lens = torch.IntTensor(seq_lens)

        return tokens, tokens_lens

    def __call__(self, text):
        tokens_seq = [p for p in text]
        seq = (
            ([self.bos_symbol] if self.add_bos else [])
            + tokens_seq
            + ([self.eos_symbol] if self.add_eos else [])
        )

        token_ids = [self.token2idx[token] for token in seq]
        token_lens = len(tokens_seq) + self.add_eos + self.add_bos

        return token_ids, token_lens


def get_text_token_collater(text_tokens_file: str) -> TextTokenCollator:
    text_tokens_path = Path(text_tokens_file)
    unique_tokens = SymbolTable.from_file(text_tokens_path)
    collater = TextTokenCollator(unique_tokens.symbols, add_bos=True, add_eos=True)
    token2idx = collater.token2idx
    return collater, token2idx


class phoneIDCollation:
    def __init__(self, cfg, dataset=None, symbols_dict_file=None) -> None:
        if cfg.preprocess.phone_extractor != "lexicon":
            ### get text token collator
            if symbols_dict_file is None:
                assert dataset is not None
                symbols_dict_file = os.path.join(
                    cfg.preprocess.processed_dir, dataset, cfg.preprocess.symbols_dict
                )
            self.text_token_colloator, token2idx = get_text_token_collater(
                symbols_dict_file
            )
            # # unique_tokens = SymbolTable.from_file(symbols_dict_path)
            # # text_tokenizer = TextToken(unique_tokens.symbols, add_bos=True, add_eos=True)

            # # update phone symbols dict file with pad_symbol or optional tokens (add_bos and add_eos) in TextTokenCollator
            # phone_symbol_dict = SymbolTable()
            # for s in sorted(list(set(token2idx.keys()))):
            #     phone_symbol_dict.add(s)
            # phone_symbol_dict.to_file(symbols_dict_file)

    def get_phone_id_sequence(self, cfg, phones_seq):
        if cfg.preprocess.phone_extractor == "lexicon":
            phones_seq = " ".join(phones_seq)
            sequence = text_to_sequence(phones_seq, cfg.preprocess.text_cleaners)
        else:
            sequence, seq_len = self.text_token_colloator(phones_seq)
        return sequence
