import json
import numpy as np
from collections import defaultdict


def load_and_index(idiom_path=None):
    """Load idiom list and adjacency graph, return indexed data structures."""
    if idiom_path is None:
        from .config import RLConfig
        idiom_path = RLConfig().idiom_file

    with open(idiom_path, 'r') as f:
        raw = json.load(f)

    # Extract idiom strings - handle both formats
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            idioms = [item['word'] for item in raw if 'word' in item]
        else:
            idioms = [str(item) for item in raw if item]
    else:
        raise ValueError(f"Unexpected JSON format: {type(raw)}")

    # Deduplicate
    idioms = list(dict.fromkeys(idioms))

    # Filter: keep only 4-character idioms
    n_before = len(idioms)
    idioms = [x for x in idioms if len(x) == 4]
    print(f"Loaded {n_before} unique idioms, {len(idioms)} with 4 characters")

    # === Idiom index ===
    idiom_to_id = {idiom: i for i, idiom in enumerate(idioms)}
    n_idioms = len(idioms)

    # === Char index ===
    all_chars = sorted(set(c for idiom in idioms for c in idiom))
    char_to_id = {c: i + 1 for i, c in enumerate(all_chars)}
    n_chars = len(char_to_id) + 1

    # === Idiom -> char ID lookup table ===
    idiom_chars = np.zeros((n_idioms, 4), dtype=np.int32)
    for idiom, idx in idiom_to_id.items():
        for j, c in enumerate(idiom[:4]):
            idiom_chars[idx, j] = char_to_id[c]

    # === Build adjacency by head-char matching ===
    # Group idioms by head character for efficient lookup
    by_head = defaultdict(list)
    for idiom, idx in idiom_to_id.items():
        by_head[idiom[0]].append(idx)

    adj_list = []
    for i, idiom in enumerate(idioms):
        tail_char = idiom[-1]
        successors = [s for s in by_head.get(tail_char, []) if s != i]
        adj_list.append(np.array(successors, dtype=np.int32))

    # === Stats ===
    degrees = [len(a) for a in adj_list]
    print(f"Before pruning - Idioms: {n_idioms}")
    print(f"Chars: {n_chars} (incl PAD)")
    print(f"Out-degree - mean: {np.mean(degrees):.1f}, "
          f"median: {np.median(degrees):.0f}, "
          f"max: {max(degrees)}, "
          f"zero-out-degree: {sum(1 for d in degrees if d == 0)}")

    # === Prune dead-end nodes (iterative) ===
    result = prune_dead_ends(idioms, adj_list, idiom_chars)
    idioms = result['idioms']
    adj_list = result['adj_list']
    idiom_chars = result['idiom_chars']
    n_idioms = result['n_idioms']
    idiom_to_id = result['idiom_to_id']
    char_to_id = result['char_to_id']
    n_chars = result['n_chars']

    degrees = [len(a) for a in adj_list]
    print(f"After pruning - Idioms: {n_idioms}")
    print(f"Zero-out-degree remaining: {sum(1 for d in degrees if d == 0)}")

    return {
        'idioms': idioms,
        'idiom_to_id': idiom_to_id,
        'char_to_id': char_to_id,
        'n_idioms': n_idioms,
        'n_chars': n_chars,
        'idiom_chars': idiom_chars,
        'adj_list': adj_list,
    }


def prune_dead_ends(idioms, adj_list, idiom_chars):
    """Iteratively remove all nodes with out-degree 0.

    Mirrors Go code's PruneDeadEnds: repeatedly removes nodes that have
    no valid successors, until every remaining node has at least one
    valid successor. This removes 3,394 iterative dead-end nodes from
    the 29,502-node exact-character graph used by the experiments.

    Returns a new data dict with pruned idioms, adj_list, idiom_chars,
    idiom_to_id, char_to_id, n_idioms, n_chars.
    """
    n = len(idioms)
    valid = np.ones(n, dtype=bool)

    while True:
        changed = False
        for u in range(n):
            if not valid[u]:
                continue
            # Count valid successors
            successors = adj_list[u]
            out_deg = int(valid[successors].sum()) if len(successors) > 0 else 0
            if out_deg == 0:
                valid[u] = False
                changed = True
        if not changed:
            break

    removed = n - valid.sum()
    print(f"  Pruning: removed {removed} iterative dead-end nodes, "
          f"{valid.sum()} nodes remaining")

    if removed == 0:
        return {
            'idioms': idioms, 'adj_list': adj_list,
            'idiom_chars': idiom_chars, 'idiom_to_id': {w: i for i, w in enumerate(idioms)},
            'n_idioms': n, 'n_chars': idiom_chars.max() + 1,
            'char_to_id': {c: i + 1 for i, c in enumerate(
                sorted(set(c for idiom in idioms for c in idiom)))},
        }

    # Reindex
    old_to_new = np.full(n, -1, dtype=np.int32)
    new_idx = 0
    for u in range(n):
        if valid[u]:
            old_to_new[u] = new_idx
            new_idx += 1

    new_n = int(valid.sum())
    new_idioms = []
    new_adj = []
    new_idiom_chars = np.zeros((new_n, 4), dtype=np.int32)

    for u in range(n):
        if valid[u]:
            nid = old_to_new[u]
            new_idioms.append(idioms[u])
            new_idiom_chars[nid] = idiom_chars[u]
            # Filter successors to only keep valid ones
            succs = adj_list[u]
            valid_succs = succs[valid[succs]]
            new_adj.append(np.array([old_to_new[v] for v in valid_succs],
                                    dtype=np.int32))

    # Rebuild char index
    new_all_chars = sorted(set(c for idiom in new_idioms for c in idiom))
    new_char_to_id = {c: i + 1 for i, c in enumerate(new_all_chars)}
    new_n_chars = len(new_char_to_id) + 1

    # Remap idiom_chars to new char IDs
    for idx, idiom in enumerate(new_idioms):
        for j, c in enumerate(idiom):
            new_idiom_chars[idx, j] = new_char_to_id[c]

    new_idiom_to_id = {idiom: i for i, idiom in enumerate(new_idioms)}

    return {
        'idioms': new_idioms,
        'adj_list': new_adj,
        'idiom_chars': new_idiom_chars,
        'idiom_to_id': new_idiom_to_id,
        'char_to_id': new_char_to_id,
        'n_idioms': new_n,
        'n_chars': new_n_chars,
    }


def validate_data(data):
    """Data integrity checks. Run before training."""
    adj_list = data['adj_list']
    n_idioms = data['n_idioms']
    idiom_chars = data['idiom_chars']
    idioms = data['idioms']

    # 1. All successor IDs in valid range
    for i, succs in enumerate(adj_list):
        assert all(0 <= s < n_idioms for s in succs), \
            f"Idiom {i} has out-of-range successor"

    # 2. No self-loops
    for i, succs in enumerate(adj_list):
        assert i not in succs, f"Idiom {i} has self-loop"

    # 3. Char ID table has no zero (PAD) entries
    assert idiom_chars.min() > 0, "Char ID 0 (PAD) found in real idioms"

    # 4. All idioms are 4 characters
    assert idiom_chars.shape[1] == 4

    # 5. Spot-check edge legality
    sample_idx = np.random.choice(n_idioms, min(100, n_idioms), replace=False)
    for i in sample_idx:
        for s in adj_list[i]:
            # tail char of source == head char of target
            assert idioms[i][-1] == idioms[s][0], \
                f"Edge {idioms[i]} -> {idioms[s]} violates chain rule"

    print("Data validation ALL PASSED")
