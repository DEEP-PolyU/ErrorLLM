import os
import re
import string
import sqlite3
import difflib
import functools
import torch
import torch.nn.functional as F
from itertools import combinations
from typing import List, Tuple, Dict, Optional, Any, Set

from rapidfuzz import fuzz
import stanza
from nltk.corpus import stopwords as nltk_stopwords

_stopwords = {
    'who', 'ourselves', ...
}
_commonwords = {"no", "yes", "many"}
_punctuation = set(string.punctuation)

MAX_CELL_NUM = 4


def _is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except Exception:
        return False


def _is_stopword(s: str) -> bool:
    return s.strip() in _stopwords


def _is_commonword(s: str) -> bool:
    return s.strip() in _commonwords


def _is_common_db_term(s: str) -> bool:
    return s.strip() in {"id"}


def _is_span_separator(c: str) -> bool:
    return c in "'\"()`,.?! "


def _split_chars(s: str) -> List[str]:
    return [c.lower() for c in s.strip()]


def _prefix_match(s1: str, s2: str) -> bool:
    i, j = 0, 0
    for i in range(len(s1)):
        if not _is_span_separator(s1[i]):
            break
    for j in range(len(s2)):
        if not _is_span_separator(s2[j]):
            break
    if i < len(s1) and j < len(s2):
        return s1[i] == s2[j]
    elif i >= len(s1) and j >= len(s2):
        return True
    return False


def _get_effective_match_source(s, start, end):
    _start = -1
    for i in range(start, start - 2, -1):
        if i < 0:
            _start = i + 1
            break
        if _is_span_separator(s[i]):
            _start = i
            break
    if _start < 0:
        return None
    _end = -1
    for i in range(end - 1, end + 3):
        if i >= len(s):
            _end = i - 1
            break
        if _is_span_separator(s[i]):
            _end = i
            break
    if _end < 0:
        return None
    while _start < len(s) and _is_span_separator(s[_start]):
        _start += 1
    while _end >= 0 and _is_span_separator(s[_end]):
        _end -= 1
    if _start > _end:
        return None
    return (_start, _end - _start + 1)


def _get_matched_entries(s, field_values, m_theta=0.85, s_theta=0.85):
    if not field_values:
        return None
    n_grams = _split_chars(s) if isinstance(s, str) else s
    matched = {}
    for fv in field_values:
        if not isinstance(fv, str):
            continue
        fv_tokens = _split_chars(fv)
        sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
        match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))
        if match.size > 0:
            source_match = _get_effective_match_source(
                n_grams, match.a, match.a + match.size)
            if source_match and source_match[1] > 1:
                match_str = fv[match.b: match.b + match.size]
                source_match_str = s[source_match[0]:
                                     source_match[0] + source_match[1]]
                c_match = match_str.lower().strip()
                c_source = source_match_str.lower().strip()
                c_fv = fv.lower().strip()
                if (c_match and not _is_number(c_match)
                        and not _is_common_db_term(c_match)):
                    if (_is_stopword(c_match) or _is_stopword(c_source)
                            or _is_stopword(c_fv)):
                        continue
                    if c_source.endswith(c_match + "'s"):
                        score = 1.0
                    else:
                        score = (fuzz.ratio(c_fv, c_source) / 100
                                 if _prefix_match(c_fv, c_source) else 0)
                    if ((_is_commonword(c_match) or _is_commonword(c_source)
                            or _is_commonword(c_fv)) and score < 1):
                        continue
                    if score >= m_theta and score >= s_theta:
                        if fv.isupper() and score * score < 1:
                            continue
                        matched[match_str] = (
                            fv, source_match_str, score, score, match.size)
    if not matched:
        return None
    return sorted(matched.items(),
                  key=lambda x: (1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]),
                  reverse=True)


@functools.lru_cache(maxsize=1000, typed=False)
def _get_column_picklist(table_name: str, column_name: str,
                         db_path: str) -> list:
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = bytes
        cursor = conn.cursor()
        cursor.execute(f'SELECT DISTINCT "{column_name}" FROM "{table_name}"')
        picklist = set()
        for x in cursor.fetchall():
            if isinstance(x[0], str):
                picklist.add(x[0].encode("utf-8"))
            elif isinstance(x[0], bytes):
                try:
                    picklist.add(x[0].decode("utf-8"))
                except UnicodeDecodeError:
                    picklist.add(x[0].decode("latin-1"))
            else:
                picklist.add(x[0])
        conn.close()
        return list(picklist)
    except Exception:
        return []


def get_database_value_matches(question: str, table_name: str, column_name: str,
                               db_path: str,
                               threshold: float = 0.85) -> List[str]:
    picklist = _get_column_picklist(table_name, column_name, db_path)
    matches = []
    if picklist and isinstance(picklist[0], str):
        matched = _get_matched_entries(
            question, picklist, m_theta=threshold, s_theta=threshold)
        if matched:
            for _ms, (fv, _sms, mscore, sscore, _msize) in matched:
                if "name" in column_name.lower() and mscore * sscore < 1:
                    continue
                if table_name.lower() != "sqlite_sequence":
                    matches.append(fv)
                    if len(matches) >= MAX_CELL_NUM:
                        break
    return matches


class SchemaLinker:

    def __init__(
        self,
        encoder,
        tokenizer,
        db_dir: str,
        use_gpu: bool = False,
        similarity_threshold: float = 0.3,
    ):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.db_dir = db_dir
        self.similarity_threshold = similarity_threshold
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

        self.stopwords = (
            _stopwords
            | set(nltk_stopwords.words("english"))
            | _punctuation
        ) - {'no'}

        stanza_gpu = torch.cuda.is_available() and use_gpu
        self.nlp = stanza.Pipeline(
            'en', processors='tokenize,pos,lemma',
            use_gpu=stanza_gpu, download_method=None,
        )
        self.word_level_tokenizer = lambda x: [
            w.lemma.lower() for s in self.nlp(x).sentences for w in s.words
        ]

        self._schema_cache: Dict[str, Tuple[List[List[str]],
                                             List[List[str]]]] = {}

    def preprocess_schema(
        self, table_meta: dict
    ) -> Tuple[List[List[str]], List[List[str]]]:
        db_id = table_meta.get("db_id", "")
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        table_names = [t.lower() for t in table_meta["table_names_original"]]
        column_names = [c.lower() for _, c in table_meta["column_names"]]

        table_toks = [self.word_level_tokenizer(name) for name in table_names]
        column_toks = [self.word_level_tokenizer(name) for name in column_names]

        self._schema_cache[db_id] = (table_toks, column_toks)
        return table_toks, column_toks

    def _value_linking(
        self,
        question: str,
        table_meta: dict,
        db_path: str,
        target_column_indices: Optional[Set[int]] = None,
    ) -> Dict[str, List[str]]:
        matched_values: Dict[str, List[str]] = {}
        column_names_original = table_meta["column_names_original"]
        table_names_original = table_meta["table_names_original"]

        for cid, (tab_id, col_name) in enumerate(column_names_original):
            if tab_id < 0:
                continue
            if (target_column_indices is not None
                    and cid not in target_column_indices):
                continue
            tab_name = table_names_original[tab_id]
            values = get_database_value_matches(
                question, tab_name, col_name, db_path)
            if values:
                matched_values[str(cid)] = values

        return matched_values

    def _extract_spans_with_offsets(
        self,
        question_toks: List[str],
        offset_mapping: List[Tuple[int, int]],
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        n = len(question_toks)
        if n == 0:
            return []

        def normalize_toks(toks):
            return self.tokenizer.convert_tokens_to_string(toks).lower().strip()

        def _filter(x):
            start, end = x
            length = end - start
            if length < 1 or length > 15:
                return False
            first_tok = question_toks[start]
            last_tok = question_toks[end - 1]
            
            if first_tok in ('Ġ', '▁') or last_tok in ('Ġ', '▁'):
                return False
            if first_tok.startswith('#'):
                return False
            return True

        indexes = sorted(
            filter(_filter, combinations(range(n + 1), 2)),
            key=lambda x: x[1] - x[0],
        )

        results = []
        for (start, end) in indexes:
            phrase = normalize_toks(question_toks[start:end])
            if phrase in self.stopwords or phrase in string.punctuation:
                continue
            if not phrase.strip():
                continue
            char_start = offset_mapping[start][0]
            char_end = offset_mapping[end - 1][1]
            results.append(((start, end), (char_start, char_end), phrase))

        return results

    def build_linking(
        self,
        question: str,
        table_meta: dict,
        db_path: Optional[str] = None,
        target_table_idxs: Optional[Set[int]] = None,
        target_column_idxs: Optional[Set[int]] = None,
    ) -> Tuple[List[dict], List[dict]]:
        from schema_utils import build_column_id

        table_names_original = table_meta["table_names_original"]
        column_names_original = table_meta["column_names_original"]
        t_num = len(table_names_original)
        c_num = len(column_names_original)

        table_toks, column_toks = self.preprocess_schema(table_meta)

        bridge_values: Dict[str, List[str]] = {}
        if db_path:
            bridge_values = self._value_linking(
                question, table_meta, db_path,
                target_column_indices=target_column_idxs,
            )

        enc = self.tokenizer(
            question,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        q_toks_check = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])
        offset_mapping = enc["offset_mapping"]

        q_len = len(q_toks_check)
        if q_len == 0:
            return [], []

        q_embeds, s_embeds, q_toks = self.encoder.encode(
            tokenizer=self.tokenizer,
            question=question,
            table_meta=table_meta,
            bridge_values=bridge_values,
            table_toks=table_toks,
            column_toks=column_toks,
            stopwords=self.stopwords,
            device=self.device,
        )

        assert len(q_toks) == q_len, (
            f"Tokenization mismatch: {len(q_toks)} vs {q_len}")

        spans = self._extract_spans_with_offsets(q_toks, offset_mapping)
        if not spans:
            return [], []

        actual_s_num = s_embeds.shape[0]
        actual_t_num = min(t_num, actual_s_num)
        actual_c_num = max(0, actual_s_num - t_num)

        target_mask = []
        for i in range(actual_t_num):
            if target_table_idxs is None or i in target_table_idxs:
                target_mask.append(True)
            else:
                target_mask.append(False)
        for i in range(actual_c_num):
            if target_column_idxs is None or i in target_column_idxs:
                target_mask.append(True)
            else:
                target_mask.append(False)
        target_mask_t = torch.tensor(target_mask, dtype=torch.bool)

        q_embeds_norm = F.normalize(q_embeds, p=2, dim=-1)
        s_embeds_norm = F.normalize(s_embeds, p=2, dim=-1)

        span_embeds = []
        for (tok_start, tok_end), _, _ in spans:
            span_embed = q_embeds_norm[tok_start:tok_end].mean(dim=0)
            span_embeds.append(span_embed)
        span_embeds_t = torch.stack(span_embeds)

        similarity = torch.mm(span_embeds_t, s_embeds_norm.t())

        similarity[:, ~target_mask_t] = -float('inf')

        all_matches: List[Tuple[
            Tuple[int, int], str, str, int]] = []

        for span_idx, ((tok_start, tok_end), (char_start, char_end), phrase) in enumerate(spans):

            table_sims = similarity[span_idx, :actual_t_num]
            
            col_sims = similarity[span_idx, actual_t_num:]

            for tidx in range(actual_t_num):
                if (table_sims[tidx] > self.similarity_threshold
                        and (target_table_idxs is None
                             or tidx in target_table_idxs)):
                    all_matches.append((
                        (char_start, char_end), phrase, 'table', tidx))

            for cidx in range(1, actual_c_num):
                if (col_sims[cidx] > self.similarity_threshold
                        and (target_column_idxs is None
                             or cidx in target_column_idxs)):
                    all_matches.append((
                        (char_start, char_end), phrase, 'column', cidx))

        if not all_matches:
            return [], []

        span_groups: Dict[Tuple[int, int], Dict] = {}
        for char_span, phrase, schema_type, schema_idx in all_matches:
            key = char_span
            if key not in span_groups:
                label = question[char_span[0]:char_span[1]]
                span_groups[key] = {
                    'label': label,
                    'phrase': phrase,
                    'targets': [],
                }
            span_groups[key]['targets'].append((schema_type, schema_idx))

        q_nodes = []
        q_edges = []
        ent_counter = 0

        for char_span in sorted(span_groups.keys()):
            info = span_groups[char_span]
            ent_counter += 1
            ent_id = f"q_ent_{ent_counter}"

            q_nodes.append({
                "id": ent_id,
                "type": "question_entity",
                "label": info['label'],
            })

            seen_targets: Set[Tuple[str, str]] = set()
            for schema_type, schema_idx in info['targets']:
                if schema_type == 'table':
                    target_id = table_names_original[schema_idx].lower()
                    rel = "relates_to_table"
                else:
                    tid = column_names_original[schema_idx][0]
                    cname = column_names_original[schema_idx][1]
                    tname = table_names_original[tid]
                    target_id = build_column_id(tname, cname)
                    rel = "relates_to_column"

                if (target_id, rel) not in seen_targets:
                    seen_targets.add((target_id, rel))
                    q_edges.append({
                        "src": ent_id,
                        "tgt": target_id,
                        "rel": rel,
                    })

        return q_nodes, q_edges
