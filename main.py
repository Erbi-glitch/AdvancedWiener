import os
import time
import gzip
import tarfile
import argparse
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple, Any, Deque, Dict
from collections import deque, defaultdict

import networkx as nx
import matplotlib.pyplot as plt

from wiener import WienerGrowingUnweighted


# --------------------------- Robust byte reading utilities ---------------------------

def _is_gzip_file(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def _iter_lines_from_path_bytes(path: str) -> Iterable[bytes]:
    opener = gzip.open if _is_gzip_file(path) else open
    with opener(path, "rb") as f:
        for raw in f:
            yield raw


def _looks_like_edgelist_name(name: str) -> bool:
    low = name.lower()
    return low.endswith((".txt", ".edges", ".edgelist", ".tsv", ".csv", ".txt.gz", ".edges.gz"))


def _choose_tar_member(tf: tarfile.TarFile, member_name: Optional[str]) -> tarfile.TarInfo:
    members = [m for m in tf.getmembers() if m.isfile()]
    if not members:
        raise FileNotFoundError("tar.gz contains no files")

    if member_name:
        for m in members:
            if m.name == member_name:
                return m
        raise FileNotFoundError(f"member not found in archive: {member_name}")

    #выбор: предпочтение отдается файлам, похожим на edgelist, выбирается самый большой файл, при равенстве значений решающим является имя.
    candidates = [m for m in members if _looks_like_edgelist_name(m.name)]
    if not candidates:
        candidates = members

    candidates.sort(key=lambda m: (-int(getattr(m, "size", 0)), m.name))
    return candidates[0]


def _iter_lines_from_tar_gz_bytes(path_tar_gz: str, member_name: Optional[str]) -> Iterable[bytes]:
    # Функция предотвращает ошибку UnicodeDecodeError для имен элементов, не использующих кодировку UTF8.
    with tarfile.open(path_tar_gz, mode="r:gz", errors="surrogateescape") as tf:
        chosen = _choose_tar_member(tf, member_name)
        fobj = tf.extractfile(chosen)
        if fobj is None:
            raise IOError(f"cannot extract member: {chosen.name}")

        if chosen.name.lower().endswith(".gz"):
            with gzip.GzipFile(fileobj=fobj, mode="rb") as gz:
                for raw in gz:
                    yield raw
        else:
            for raw in fobj:
                yield raw


def load_edgelist_any(path: str, tar_member: Optional[str] = None) -> nx.Graph:
    #Загрузка неориентированного графа

    G = nx.Graph()
    low = path.lower()

    if low.endswith(".tar.gz"):
        line_iter = _iter_lines_from_tar_gz_bytes(path, tar_member)
    else:
        line_iter = _iter_lines_from_path_bytes(path)

    for raw in line_iter:
        raw = raw.strip()
        if not raw or raw.startswith(b"#"):
            continue
        parts = raw.split()
        if len(parts) < 2:
            continue

        u_raw, v_raw = parts[0], parts[1]
        try:
            u = int(u_raw)
            v = int(v_raw)
        except ValueError:
            u = u_raw
            v = v_raw

        if u != v:
            G.add_edge(u, v)

    return G


# --------------------------- Deterministic component + deterministic BFS tree ---------------------------

def _repr_key(x: Any) -> str:
    return repr(x)


def _norm_edge(a: Any, b: Any) -> Tuple[Any, Any]:
    return (a, b) if _repr_key(a) <= _repr_key(b) else (b, a)


def largest_component_deterministic(G: nx.Graph) -> nx.Graph:
    
    #Выберите наибольшую связную компоненту
    #В случае равенства результатов: компонент с наименьшим значением min(repr(node))

    if G.number_of_nodes() == 0:
        return G

    if nx.is_connected(G):
        return G

    comps = list(nx.connected_components(G))
    comps.sort(key=lambda c: (-len(c), min(_repr_key(x) for x in c)))
    return G.subgraph(comps[0]).copy()


def bfs_tree_edges_sorted(G: nx.Graph, root: Any) -> List[Tuple[Any, Any]]:

    #Детерминированный поиск ребер в ширину (BFS)
    #Возвращает ребра (родитель, потомок) в порядке обнаружения

    seen: Set[Any] = {root}
    q: Deque[Any] = deque([root])
    tree_edges: List[Tuple[Any, Any]] = []

    while q:
        u = q.popleft()
        for v in sorted(G[u], key=_repr_key):
            if v in seen:
                continue
            seen.add(v)
            q.append(v)
            tree_edges.append((u, v))

    return tree_edges


# --------------------------- Randomized execution schedule (leaf vs edge) ---------------------------

@dataclass(frozen=True)
class Op:
    kind: str   # "leaf" | "edge"
    u: Any
    v: Any


def build_growth_structures(G: nx.Graph) -> Tuple[Any, Dict[Any, Deque[Any]], List[Tuple[Any, Any]]]:

    #Из связного статического графа G:
    #выбраем корень (узел с минимальным представлением)
    #строим ребра дерева BFS (детерминированное)
    #затем children_by_parent: родитель -> deque(children)
    #строим extra_edges: все ребра, не входящие в дерево BFS (нормализованные, отсортированные)

    root = min(G.nodes(), key=_repr_key)

    tree_edges = bfs_tree_edges_sorted(G, root)
    tree_edge_set: Set[Tuple[Any, Any]] = set(_norm_edge(a, b) for a, b in tree_edges)

    children_by_parent: Dict[Any, Deque[Any]] = defaultdict(deque)
    for p, c in tree_edges:
        children_by_parent[p].append(c)

    extra_edges: List[Tuple[Any, Any]] = []
    for a, b in G.edges():
        e = _norm_edge(a, b)
        if e not in tree_edge_set:
            extra_edges.append(e)

    extra_edges.sort(key=lambda e: (_repr_key(e[0]), _repr_key(e[1])))
    return root, children_by_parent, extra_edges


def make_randomized_ops(
    root: Any,
    children_by_parent: Dict[Any, Deque[Any]],
    extra_edges: List[Tuple[Any, Any]],
    max_ops: int,
    seed: int,
    p_edge: float,
) -> List[Op]:

    #Создаем случайную *допустимую* последовательность операций (до max_ops), чередуя
    #операции с листьями, операции с ребрами

    rng = random.Random(seed)

    active: Set[Any] = {root}

    # Родитель может заспавнить leaf
    leaf_parents: Set[Any] = set()
    if children_by_parent.get(root):
        leaf_parents.add(root)

    # Создает списки инцидентности для дополнительных ребер: node -> edge
    incident: Dict[Any, List[int]] = defaultdict(list)
    for idx, (a, b) in enumerate(extra_edges):
        incident[a].append(idx)
        incident[b].append(idx)

    used_edge: Set[int] = set()
    eligible_edges: Set[int] = set()

    def activate_edges_for_new_node(u: Any) -> None:
        for idx in incident.get(u, []):
            if idx in used_edge:
                continue
            a, b = extra_edges[idx]
            other = b if a == u else a
            if other in active:
                eligible_edges.add(idx)

    ops: List[Op] = []

    for _ in range(max_ops):
        has_leaf = len(leaf_parents) > 0
        has_edge = len(eligible_edges) > 0

        if not has_leaf and not has_edge:
            break

        # Рандомный выбор
        if has_leaf and has_edge:
            choose_edge = (rng.random() < p_edge)
        else:
            choose_edge = has_edge  # если остаются edge

        if choose_edge:
            idx = rng.choice(tuple(eligible_edges))
            eligible_edges.remove(idx)
            used_edge.add(idx)
            a, b = extra_edges[idx]
            ops.append(Op("edge", a, b))
        else:
            parent = rng.choice(tuple(leaf_parents))
            child = children_by_parent[parent].popleft()
            ops.append(Op("leaf", child, parent))

            # новый узел
            active.add(child)

            # обновляем
            if not children_by_parent[parent]:
                leaf_parents.discard(parent)
            if children_by_parent.get(child):
                leaf_parents.add(child)

            activate_edges_for_new_node(child)

    return ops


# --------------------------- Dataset file selection ---------------------------

def pick_dataset_file(script_dir: str, prefer: Optional[str]) -> str:

    #prefet -> .tar.gz -> .gz -> plain

    if prefer:
        path = prefer
        if not os.path.isabs(path):
            path = os.path.join(script_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"dataset not found: {path}")
        return path

    exts = (".tar.gz", ".txt.gz", ".edges.gz", ".txt", ".edges", ".edgelist", ".tsv", ".csv")
    candidates = []
    for name in os.listdir(script_dir):
        if name == os.path.basename(__file__):
            continue
        if name.lower().endswith(exts):
            candidates.append(name)


    def rank(name: str) -> Tuple[int, str]:
        low = name.lower()
        if low.endswith(".tar.gz"):
            r = 0
        elif low.endswith(".gz"):
            r = 1
        else:
            r = 2
        return (r, name)

    candidates.sort(key=rank)
    return os.path.join(script_dir, candidates[0])


# --------------------------- Benchmark (time-only) + plot ---------------------------

def benchmark_time_arrays(root: Any, ops: List[Op]) -> dict:
    """
    функция из библиотеки:
    - применить операцию к G_base
    - измерять время выполнения nx.wiener_index(G_base) на каждом шаге
    моя функция:
    - измерять время выполнения обработчика операции (полный поиск в ширину листа / пересчет ребра)
    """
    # функция из библиотеки
    G_base = nx.Graph()
    G_base.add_node(root)

    baseline_step_times: List[float] = []
    for op in ops:
        if op.kind == "leaf":
            G_base.add_node(op.u)
            G_base.add_edge(op.u, op.v)
        else:
            G_base.add_edge(op.u, op.v)

        ts = time.perf_counter()
        _ = nx.wiener_index(G_base)
        te = time.perf_counter()
        baseline_step_times.append(te - ts)

    baseline_cum = []
    s = 0.0
    for x in baseline_step_times:
        s += x
        baseline_cum.append(s)

    # моя функция
    inc = WienerGrowingUnweighted()
    inc.add_initial_node(root)

    inc_step_times: List[float] = []
    leaf_count = 0
    edge_count = 0

    for op in ops:
        ts = time.perf_counter()
        if op.kind == "leaf":
            inc.add_leaf(op.u, op.v)
            leaf_count += 1
        else:
            inc.add_edge(op.u, op.v)
            edge_count += 1
        te = time.perf_counter()
        inc_step_times.append(te - ts)

    inc_cum = []
    s = 0.0
    for x in inc_step_times:
        s += x
        inc_cum.append(s)

    return {
        "ops_ran": len(ops),
        "leaf_ops_ran": leaf_count,
        "edge_ops_ran": edge_count,
        "baseline_cum_times": baseline_cum,
        "inc_cum_times": inc_cum,
    }


def save_time_plot_png(stats: dict, out_path: str) -> None:
    n = min(len(stats["baseline_cum_times"]), len(stats["inc_cum_times"]))
    x = list(range(1, n + 1))

    plt.figure()
    plt.plot(x, stats["baseline_cum_times"][:n], label="NetworkX cumulative")
    plt.plot(x, stats["inc_cum_times"][:n], label="Incremental cumulative")
    plt.xlabel("Step")
    plt.ylabel("Cumulative time (s)")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------- Main ---------------------------

def main() -> None:
    #аргументы командной строки
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None)
    ap.add_argument("--tar-member", default=None)
    ap.add_argument("--max-ops", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-edge", type=float, default=0.2)
    ap.add_argument("--time-plot-out", default="time_plot.png")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = pick_dataset_file(script_dir, args.dataset)

    print(f"Dataset file: {dataset_path}")

    G = load_edgelist_any(dataset_path, tar_member=args.tar_member)
    print(f"Loaded graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    H = largest_component_deterministic(G)
    print(f"Largest component: nodes={H.number_of_nodes()}, edges={H.number_of_edges()}")

    if H.number_of_nodes() == 0:
        print("Empty graph after component selection.")
        return

    root, children_by_parent, extra_edges = build_growth_structures(H)

    leaf_total = H.number_of_nodes() - 1
    edge_total = H.number_of_edges() - leaf_total
    print(f"Planned totals in component: leaf_total={leaf_total}, edge_total={edge_total}")

    ops = make_randomized_ops(
        root=root,
        children_by_parent=children_by_parent,
        extra_edges=extra_edges,
        max_ops=args.max_ops,
        seed=args.seed,
        p_edge=args.p_edge,
    )

    stats = benchmark_time_arrays(root, ops)
    print(f"ops_ran: {stats['ops_ran']}")
    print(f"leaf_ops_ran: {stats['leaf_ops_ran']}")
    print(f"edge_ops_ran: {stats['edge_ops_ran']}")

    save_time_plot_png(stats, args.time_plot_out)
    print(f"Saved time plot to: {args.time_plot_out}")


if __name__ == "__main__":
    main()
