import math
import copy
import string as string_module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from itertools import combinations, product as iterproduct
from typing import List, Dict, Tuple, Set


MAX_RELATIVE_DIST = 4
ENCODER_RELATIONS = (
    ['padding-padding']
    + [f'question-question-dist{i:d}'
       for i in range(-MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)]
    + ['question-question-previous', 'question-question-after']
    + ['table-table-identity', 'table-table-fk',
       'table-table-fkr', 'table-table-fkb']
    + ['column-column-identity', 'column-column-sametable',
       'column-column-fk', 'column-column-fkr']
    + ['table-column-pk', 'column-table-pk',
       'table-column-has', 'column-table-has']
    + ['question-column-exactmatch', 'question-column-partialmatch',
       'question-column-nomatch', 'question-column-valuematch',
       'column-question-exactmatch', 'column-question-partialmatch',
       'column-question-nomatch', 'column-question-valuematch']
    + ['question-table-exactmatch', 'question-table-partialmatch',
       'question-table-nomatch',
       'table-question-exactmatch', 'table-question-partialmatch',
       'table-question-nomatch']
    + ['question-question-generic', 'table-table-generic',
       'column-column-generic', 'table-column-generic',
       'column-table-generic']
)
_R = {name: idx for idx, name in enumerate(ENCODER_RELATIONS)}

def build_question_relation(q_len: int) -> np.ndarray:
    dtype = np.int64
    if q_len <= MAX_RELATIVE_DIST + 1:
        dist_vec = [_R[f'question-question-dist{i:d}']
                    for i in range(-MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)]
        starting = MAX_RELATIVE_DIST
    else:
        dist_vec = (
            [_R['question-question-generic']] * (q_len - MAX_RELATIVE_DIST - 1)
            + [_R[f'question-question-dist{i:d}']
               for i in range(-MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)]
            + [_R['question-question-generic']] * (q_len - MAX_RELATIVE_DIST - 1)
        )
        starting = q_len - 1
    return np.array(
        [dist_vec[starting - i: starting - i + q_len] for i in range(q_len)],
        dtype=dtype,
    )


def build_schema_relation(table_meta: dict) -> np.ndarray:
    dtype = np.int64
    t_num = len(table_meta['table_names_original'])
    c_num = len(table_meta['column_names'])
    column2table = [x[0] for x in table_meta['column_names']]

    tab_mat = np.full((t_num, t_num), _R['table-table-generic'], dtype=dtype)
    fk_pairs = table_meta.get('foreign_keys', [])
    table_fks = set()
    for c1, c2 in fk_pairs:
        table_fks.add((column2table[c1], column2table[c2]))
    for (t1, t2) in table_fks:
        if (t2, t1) in table_fks:
            tab_mat[t1, t2] = tab_mat[t2, t1] = _R['table-table-fkb']
        else:
            tab_mat[t1, t2] = _R['table-table-fk']
            tab_mat[t2, t1] = _R['table-table-fkr']
    np.fill_diagonal(tab_mat, _R['table-table-identity'])

    col_mat = np.full((c_num, c_num), _R['column-column-generic'], dtype=dtype)
    for t_idx in range(t_num):
        col_ids = [i for i, t in enumerate(column2table) if t == t_idx]
        for c1, c2 in iterproduct(col_ids, col_ids):
            col_mat[c1, c2] = _R['column-column-sametable']
    np.fill_diagonal(col_mat, _R['column-column-identity'])
    if fk_pairs:
        for c1, c2 in fk_pairs:
            col_mat[c1, c2] = _R['column-column-fk']
            col_mat[c2, c1] = _R['column-column-fkr']
    col_mat[0, :] = _R['column-column-generic']
    col_mat[:, 0] = _R['column-column-generic']
    col_mat[0, 0] = _R['column-column-identity']

    tab_col_mat = np.full((t_num, c_num), _R['table-column-generic'], dtype=dtype)
    col_tab_mat = np.full((c_num, t_num), _R['column-table-generic'], dtype=dtype)
    for cid in range(1, c_num):
        tid = column2table[cid]
        if tid >= 0:
            col_tab_mat[cid, tid] = _R['column-table-has']
            tab_col_mat[tid, cid] = _R['table-column-has']
    pks = table_meta.get('primary_keys', [])
    for pk in pks:
        if isinstance(pk, list):
            for p in pk:
                tid = column2table[p]
                col_tab_mat[p, tid] = _R['column-table-pk']
                tab_col_mat[tid, p] = _R['table-column-pk']
        else:
            tid = column2table[pk]
            col_tab_mat[pk, tid] = _R['column-table-pk']
            tab_col_mat[tid, pk] = _R['table-column-pk']
    col_tab_mat[0, :] = _R['column-table-has']
    tab_col_mat[:, 0] = _R['table-column-has']

    return np.concatenate([
        np.concatenate([tab_mat, tab_col_mat], axis=1),
        np.concatenate([col_tab_mat, col_mat], axis=1),
    ], axis=0)


def build_question_schema_relations(
    question_toks: List[str],
    table_names: List[str],
    column_names: List[str],
    table_toks: List[List[str]],
    column_toks: List[List[str]],
    bridge_values: Dict[str, List[str]],
    tokenizer,
    stopwords: Set[str],
) -> Tuple[np.ndarray, np.ndarray]:
    dtype = np.int64
    q_num = len(question_toks)
    t_num = len(table_names)
    c_num = len(column_names)

    def normalize_toks(toks):
        return tokenizer.convert_tokens_to_string(toks).lower().strip()

    def _filter(x):
        return (1 <= x[1] - x[0] <= 15
                and question_toks[x[0]] not in ['Ġ', '▁']
                and question_toks[x[1] - 1] not in ['Ġ', '▁']
                and not question_toks[x[0]].startswith('#'))

    indexes = sorted(
        filter(_filter, combinations(range(q_num + 1), 2)),
        key=lambda x: x[1] - x[0],
    )
    pairs = [(p, normalize_toks(question_toks[p[0]:p[1]])) for p in indexes]
    pairs = [(p, ph) for p, ph in pairs
             if ph not in stopwords and ph not in string_module.punctuation
             and ph.strip()]

    qt_mat = np.full((q_num, t_num), _R['question-table-nomatch'], dtype=dtype)
    tq_mat = np.full((t_num, q_num), _R['table-question-nomatch'], dtype=dtype)
    for sid, sname in enumerate(table_names):
        max_len = len(tokenizer.tokenize(sname))
        stoks = table_toks[sid]
        for (start, end), phrase in pairs:
            if end - start > max_len:
                break
            if (phrase == sname
                    or (sname in phrase
                        and len(sname) >= 2 * len(phrase) / 3)):
                qt_mat[range(start, end), sid] = _R['question-table-exactmatch']
                tq_mat[sid, range(start, end)] = _R['table-question-exactmatch']
            elif phrase in sname.split(' ') or phrase in stoks:
                qt_mat[range(start, end), sid] = _R['question-table-partialmatch']
                tq_mat[sid, range(start, end)] = _R['table-question-partialmatch']

    qc_mat = np.full((q_num, c_num), _R['question-column-nomatch'], dtype=dtype)
    cq_mat = np.full((c_num, q_num), _R['column-question-nomatch'], dtype=dtype)

    if bridge_values:
        for cid_str, cells in bridge_values.items():
            cid = int(cid_str)
            for c in cells:
                toks = tokenizer.tokenize(c)
                l = len(toks)
                nc = normalize_toks(toks)
                for (start, end), phrase in pairs:
                    if l - 2 <= end - start <= l + 2 and phrase in nc:
                        qc_mat[range(start, end), cid] = \
                            _R['question-column-valuematch']
                        cq_mat[cid, range(start, end)] = \
                            _R['column-question-valuematch']

    for sid, sname in enumerate(column_names):
        if sid == 0:
            continue
        max_len = len(tokenizer.tokenize(sname))
        stoks = column_toks[sid]
        for (start, end), phrase in pairs:
            if end - start > max_len:
                break
            if (phrase == sname
                    or (sname in phrase
                        and len(sname) >= 2 * len(phrase) / 3)):
                qc_mat[range(start, end), sid] = \
                    _R['question-column-exactmatch']
                cq_mat[sid, range(start, end)] = \
                    _R['column-question-exactmatch']
            elif phrase in sname.split(' ') or phrase in stoks:
                qc_mat[range(start, end), sid] = \
                    _R['question-column-partialmatch']
                cq_mat[sid, range(start, end)] = \
                    _R['column-question-partialmatch']

    qs_mat = np.concatenate([qt_mat, qc_mat], axis=1)
    sq_mat = np.concatenate([tq_mat, cq_mat], axis=0)
    return qs_mat, sq_mat


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def rnn_wrapper(encoder, inputs, lens, cell='lstm'):
    sorted_lens, sort_key = torch.sort(lens, descending=True)
    nonzero_num = torch.sum(sorted_lens > 0).item()
    total_num = sorted_lens.size(0)
    sort_key_nz = sort_key[:nonzero_num]
    sorted_inputs = torch.index_select(inputs, dim=0, index=sort_key_nz)
    packed = rnn_utils.pack_padded_sequence(
        sorted_inputs, sorted_lens[:nonzero_num].tolist(), batch_first=True)
    packed_out, sorted_h = encoder(packed)
    sorted_out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
    if cell.upper() == 'LSTM':
        sorted_h, sorted_c = sorted_h
    out_shape = list(sorted_out.size())
    out_shape[0] = total_num
    h_shape = list(sorted_h.size())
    h_shape[1] = total_num
    idx_o = sort_key_nz.unsqueeze(-1).unsqueeze(-1).repeat(1, *out_shape[1:])
    idx_h = sort_key_nz.unsqueeze(0).unsqueeze(-1).repeat(
        h_shape[0], 1, h_shape[-1])
    out = sorted_out.new_zeros(*out_shape).scatter_(0, idx_o, sorted_out)
    h = sorted_h.new_zeros(*h_shape).scatter_(1, idx_h, sorted_h)
    if cell.upper() == 'LSTM':
        c = sorted_c.new_zeros(*h_shape).scatter_(1, idx_h, sorted_c)
        return out, (h.contiguous(), c.contiguous())
    return out, h.contiguous()


class FFN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size * 4, input_size),
        )
        self.layernorm = nn.LayerNorm(input_size)

    def forward(self, inputs):
        return self.layernorm(inputs + self.feedforward(inputs))


class RGATLayer(nn.Module):
    def __init__(self, hidden_size=256, num_heads=8, dropout=0.2):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale_factor = math.sqrt(hidden_size // num_heads)
        self.concat_affine = nn.Linear(hidden_size, hidden_size, bias=True)
        self.feedforward = FFN(hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, inputs, mask, rel_k, rel_v):
        bs, l = inputs.size(0), inputs.size(1)
        q = self.query(self.dropout_layer(inputs))
        k = self.key(self.dropout_layer(inputs))
        v = self.value(self.dropout_layer(inputs))
        
        q = q.view(bs, l, self.num_heads, -1).transpose(1, 2).unsqueeze(3)
        
        k = k.view(bs, l, self.num_heads, -1).transpose(1, 2) \
             .unsqueeze(2).expand(bs, self.num_heads, l, l, -1)
        v = v.view(bs, l, self.num_heads, -1).transpose(1, 2) \
             .unsqueeze(2).expand(bs, self.num_heads, l, l, -1)
        
        k, v = k + rel_k, v + rel_v
        
        e = (torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor) \
            .squeeze(-2)
        e = e.masked_fill_(mask.unsqueeze(1), -1e20)
        a = torch.softmax(e, dim=-1)
        outputs = torch.matmul(a.unsqueeze(-2), v).squeeze(-2)
        outputs = outputs.transpose(1, 2).contiguous().view(bs, l, -1)
        outputs = self.concat_affine(outputs)
        outputs = self.layernorm(inputs + outputs)
        return self.feedforward(outputs)


class RGATEncoder(nn.Module):

    def __init__(self, plm_path: str, hidden_size: int = 256,
                 num_heads: int = 8, num_layers: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        config = AutoConfig.from_pretrained(plm_path)
        self.plm = AutoModel.from_pretrained(plm_path)
        self.plm_hidden_size = config.hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.question_rnn = nn.LSTM(
            self.plm_hidden_size, hidden_size // 2,
            num_layers=1, bidirectional=True, batch_first=True)
        self.schema_rnn = nn.LSTM(
            self.plm_hidden_size, hidden_size // 2,
            num_layers=1, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=dropout)

        rn = len(ENCODER_RELATIONS)
        pad_idx = _R['padding-padding']
        self.relation_embed_k = nn.Embedding(
            rn, hidden_size // num_heads, padding_idx=pad_idx)
        self.relation_embed_v = nn.Embedding(
            rn, hidden_size // num_heads, padding_idx=pad_idx)
        gnn_module = RGATLayer(hidden_size, num_heads, dropout=dropout)
        self.gnn_layers = clones(gnn_module, num_layers)

    @torch.no_grad()
    def encode(
        self,
        tokenizer,
        question: str,
        table_meta: dict,
        bridge_values: Dict[str, List[str]],
        table_toks: List[List[str]],
        column_toks: List[List[str]],
        stopwords: Set[str],
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        self.eval()
        self.to(device)

        table_names = [t.lower() for t in table_meta['table_names']]
        column_names = [c.lower() for _, c in table_meta['column_names']]
        t_num = len(table_names)
        c_num = len(column_names)
        s_num = t_num + c_num

        q_toks = tokenizer.tokenize(question.strip())
        q_ids = tokenizer.convert_tokens_to_ids(q_toks)
        q_len = len(q_ids)
        if q_len == 0:
            hs = self.hidden_size
            return (torch.zeros(0, hs), torch.zeros(s_num, hs), q_toks)

        table_prefix = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize('table'))
        table_token_ids_list = []
        table_token_lens = []
        for tname in table_meta['table_names']:
            tids = table_prefix + tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(tname.lower()))
            table_token_ids_list.append(tids)
            table_token_lens.append(len(tids))

        column_types = table_meta.get(
            'column_types', ['text'] * c_num)
        column_token_ids_list = []
        column_token_lens = []
        for cid in range(c_num):
            ctype = column_types[cid] if cid < len(column_types) else 'text'
            cname = table_meta['column_names'][cid][1]
            tids = (tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(ctype))
                    + tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(cname.lower())))
            if str(cid) in bridge_values:
                val_span = ': ' + ' , '.join(bridge_values[str(cid)])
                tids += tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(val_span))
            column_token_ids_list.append(tids)
            column_token_lens.append(len(tids))

        schema_ids = (sum(table_token_ids_list, [])
                      + sum(column_token_ids_list, []))
        schema_token_lens = table_token_lens + column_token_lens

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        max_len = 512  

        input_ids = ([cls_id] + q_ids + [sep_id]
                     + schema_ids + [sep_id])
        total_input_len = len(input_ids)

        if total_input_len > max_len:
            q_space = 1 + q_len + 2 
            schema_budget = max_len - q_space
            if schema_budget < 10:
                q_ids = q_ids[:max_len - 20]
                q_len = len(q_ids)
                q_toks = q_toks[:q_len]
                schema_budget = max_len - (1 + q_len + 2)

            new_schema_ids = []
            new_schema_token_lens = []
            token_count = 0
            for i, stl in enumerate(schema_token_lens):
                if token_count + stl > schema_budget:
                    break
                
                if i < t_num:
                    item_toks = table_token_ids_list[i]
                else:
                    item_toks = column_token_ids_list[i - t_num]
                new_schema_ids.extend(item_toks)
                new_schema_token_lens.append(stl)
                token_count += stl

            schema_ids = new_schema_ids
            schema_token_lens = new_schema_token_lens
            s_num = len(schema_token_lens)

            input_ids = ([cls_id] + q_ids + [sep_id]
                         + schema_ids + [sep_id])
            total_input_len = len(input_ids)
        segment_ids = ([0] * (q_len + 2)
                       + [1] * (len(schema_ids) + 1))
        attn_mask = [1] * total_input_len
        plm_q_mask = ([False] + [True] * q_len
                      + [False] * (len(schema_ids) + 2))
        plm_s_mask = ([False] * (q_len + 2)
                      + [True] * len(schema_ids) + [False])

        input_ids_t = torch.tensor(
            [input_ids], dtype=torch.long, device=device)
        segment_ids_t = torch.tensor(
            [segment_ids], dtype=torch.long, device=device)
        attn_mask_t = torch.tensor(
            [attn_mask], dtype=torch.float, device=device)
        plm_q_mask_t = torch.tensor(
            [plm_q_mask], dtype=torch.bool, device=device)
        plm_s_mask_t = torch.tensor(
            [plm_s_mask], dtype=torch.bool, device=device)

        plm_out = self.plm(
            input_ids=input_ids_t,
            attention_mask=attn_mask_t,
            token_type_ids=segment_ids_t,
        )[0] 

        plm_q = plm_out.masked_select(
            plm_q_mask_t.unsqueeze(-1)).view(1, q_len, -1)
        q_lens_t = torch.tensor([q_len], dtype=torch.long, device=device)
        q_out, _ = rnn_wrapper(
            self.question_rnn, self.dropout_layer(plm_q), q_lens_t)
        q_embeds = q_out.squeeze(0)  

        plm_s = plm_out.masked_select(
            plm_s_mask_t.unsqueeze(-1))
        total_stoks = sum(schema_token_lens)
        plm_s = plm_s.view(total_stoks, -1)
        max_stl = max(schema_token_lens) if schema_token_lens else 1
        n_items = len(schema_token_lens)
        schema_padded = plm_s.new_zeros(n_items, max_stl, plm_s.size(-1))
        offset = 0
        for i, stl in enumerate(schema_token_lens):
            schema_padded[i, :stl] = plm_s[offset:offset + stl]
            offset += stl
        stl_t = torch.tensor(
            schema_token_lens, dtype=torch.long, device=device)
        _, s_hidden = rnn_wrapper(
            self.schema_rnn, self.dropout_layer(schema_padded), stl_t)
        s_embeds = s_hidden[0].transpose(0, 1).contiguous().view(
            n_items, -1)  

        qq_rel = build_question_relation(q_len)
        ss_rel_full = build_schema_relation(table_meta)
        qs_rel_full, sq_rel_full = build_question_schema_relations(
            q_toks, table_names, column_names,
            table_toks, column_toks, bridge_values,
            tokenizer, stopwords)

        n_items = len(schema_token_lens)
        ss_rel = ss_rel_full[:n_items, :n_items]
        qs_rel = qs_rel_full[:, :n_items]
        sq_rel = sq_rel_full[:n_items, :]

        total_len = q_len + n_items
        pad_val = _R['padding-padding']
        full_rel = np.full((total_len, total_len), pad_val, dtype=np.int64)
        full_rel[:q_len, :q_len] = qq_rel
        full_rel[:q_len, q_len:] = qs_rel
        full_rel[q_len:, :q_len] = sq_rel
        full_rel[q_len:, q_len:] = ss_rel
        encoder_relations = torch.tensor(
            full_rel, dtype=torch.long, device=device).unsqueeze(0)

        combined = torch.cat(
            [q_embeds, s_embeds], dim=0).unsqueeze(0)  
        rel_mask = (encoder_relations == pad_val)
        rel_k = self.relation_embed_k(encoder_relations) \
            .unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)
        rel_v = self.relation_embed_v(encoder_relations) \
            .unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)

        outputs = combined
        for layer in self.gnn_layers:
            outputs = layer(outputs, rel_mask, rel_k, rel_v)

        outputs = outputs.squeeze(0)  
        return outputs[:q_len], outputs[q_len:], q_toks
