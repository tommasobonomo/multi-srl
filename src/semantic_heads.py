from typing import List, Optional, Set, Tuple

from anytree import LevelOrderIter, Node
from wasabi import msg

from src.token_types import Sentence, TokenFeatures

# Set of Universal Dependencies 2.5 dependency relations that indicate a core-word of the sentence
# Taken from https://universaldependencies.org/u/dep/index.html
POSSIBLE_SEMANTIC_HEADS = set(
    (
        # Nominal core dependents of clausal predicates
        "nsubj",
        "obj",
        "iobj",
        # Predicate core dependents of clausal predicates
        "csubj",
        "ccomp",
        "xcomp",
        # Nominal non-core dependents
        "obl",
        "vocative",
        "expl",
        # Nominal dependents
        "nmod",
    )
)


def old_english_to_universal_semantic_head(
    role_idx: int, tokens: List[TokenFeatures], og_frame_idx: int, depth: int = 0
) -> Optional[int]:
    """
    Recursive function that finds the semantic head of a role. It assumes that the semantic
    dependency annotation is available for each token (i.e. `TokenFeatures.sem_dep` is not all `None` fields)
    We recursively search up from the given node until we find a core node.
    """
    assert tokens[role_idx].sem_dep.is_any_none() is False, "No semantic dep annotation"
    dep_tag = tokens[role_idx].sem_dep.relation.split(":")[0]  # type: ignore
    if depth > 1:
        # Stop search at one level over
        return None
    else:
        if dep_tag in POSSIBLE_SEMANTIC_HEADS and role_idx != og_frame_idx:
            return role_idx
        elif dep_tag == "ROOT" or role_idx == og_frame_idx:
            return None
        else:
            return old_english_to_universal_semantic_head(
                tokens[role_idx].sem_dep.head_idx, tokens, og_frame_idx, depth + 1  # type: ignore
            )


def match_span(
    span_indices: Set[int],
    target_tree_root: Node,
    tokens: Sentence,
) -> int:
    """
    Algorithm that, given a span with some indices and a target tree, finds the subtree in the target tree
    that is most representative of the span.
    This is done by finding the subtree with the smallest symmetric difference between the span indices
    and the subtree indices.
    """
    smallest_subtree = target_tree_root
    smallest_symmetric_difference = span_indices.symmetric_difference(
        set(token.idx for token in tokens)
    )
    for node in LevelOrderIter(target_tree_root):
        candidate_subtree_indices = set(n.idx for n in node.descendants).union(
            {node.idx}
        )
        if len(span_indices.symmetric_difference(candidate_subtree_indices)) < len(
            smallest_symmetric_difference
        ):
            smallest_subtree = node
            smallest_symmetric_difference = span_indices.symmetric_difference(
                candidate_subtree_indices
            )

    return smallest_subtree.idx


def english_to_universal_semantic_head(
    role_idx: int, tokens: List[TokenFeatures]
) -> Optional[int]:
    """
    This function takes a syntactic DEP-based SRL prediction and converts it to a semantic role.
    It assumes that both semantic and syntactic dependencies are available for each token.
    Returns the semantic head of the role.
    """
    syntactic_root_node = parse_sentence_to_dep_tree(tokens, is_semantic=False)
    syntactic_subtree_node = next(
        node for node in LevelOrderIter(syntactic_root_node) if node.idx == role_idx
    )
    span_indices = set(node.idx for node in LevelOrderIter(syntactic_subtree_node))

    semantic_root_node = parse_sentence_to_dep_tree(tokens, is_semantic=True)
    # The semantic subtree node is the smallest subtree that contains all nodes in span_indices
    return match_span(span_indices, semantic_root_node, tokens)


def universal_to_english_semantic_head(role_idx: int, tokens: Sentence) -> int:
    """
    This function takes a semantic DEP-based SRL prediction and converts it to a syntactic role.
    It assumes that both semantic and syntactic dependencies are available for each token.
    Returns the syntactic head of the role.
    """

    semantic_root_node = parse_sentence_to_dep_tree(tokens, is_semantic=True)
    semantic_subtree_node = next(
        node for node in LevelOrderIter(semantic_root_node) if node.idx == role_idx
    )
    span_indices = set(node.idx for node in LevelOrderIter(semantic_subtree_node))

    syntactic_root_node = parse_sentence_to_dep_tree(tokens, is_semantic=False)
    # The syntactic subtree node is the smallest subtree that contains all nodes in span_indices
    return match_span(span_indices, syntactic_root_node, tokens)


def span_to_dep_with_semantic_head(
    span_idx: Tuple[int, int], tokens: List[TokenFeatures]
) -> Optional[int]:
    """
    Parses a span-annotated SRL role to a dep-annotated, through dependency parsing of tree.
    `span_idx` should contain the start (inclusive) and end (exclusive) index of the span.
    """
    span_start, span_end = span_idx

    # TODO: problem if span contains multiple subtrees

    # Build set of span indices
    span_indices = set(token.idx for token in tokens[span_start:span_end])
    # Parse sentence as dep tree
    root_node = parse_sentence_to_dep_tree(tokens, is_semantic=True)
    # BFS to find root of span
    span_root_node: Node = next(
        node for node in LevelOrderIter(root_node) if node.idx in span_indices
    )

    # BFS of span subtree, stop if you find a "content" dep going down.
    dep_node = next(
        (
            node
            for node in LevelOrderIter(span_root_node)
            if tokens[node.idx].sem_dep.relation.split(":")[0]
            in POSSIBLE_SEMANTIC_HEADS
        ),
        None,
    )

    return dep_node.idx if dep_node else None


def token_roles(
    token: TokenFeatures, sentence: Sentence
) -> List[Tuple[Optional[str], str]]:
    return [
        (other_token.frame_name, frame_role.role_name)
        for other_token in sentence
        for frame_role in other_token.frame_roles
        if frame_role.role_idx == token.idx
    ]


def get_root_depth(token: TokenFeatures, is_semantic: bool, sentence: Sentence) -> int:
    if is_semantic:
        dep = token.sem_dep
    else:
        dep = token.synt_dep

    if len(dep.child_indices) == 0:
        return 0
    else:
        return 1 + max(
            get_root_depth(sentence[idx], is_semantic, sentence)
            for idx in dep.child_indices
        )


def build_nodes(
    child_indices: List[int], sentence: Sentence, is_semantic: bool
) -> List[Node]:
    """Recursively build child nodes given the child indices and the source sentence"""
    child_nodes = []
    for idx in child_indices:
        token = sentence[idx]
        dep_relation = token.sem_dep if is_semantic else token.synt_dep
        child_nodes.append(
            Node(
                token.idx,
                idx=token.idx,
                dep=dep_relation.relation,
                text=token.text,
                frame_name=token.frame_name,
                roles=str(token_roles(token, sentence)),
                children=build_nodes(dep_relation.child_indices, sentence, is_semantic),
            )
        )
    return child_nodes


def parse_sentence_to_dep_tree(sentence: Sentence, is_semantic: bool = False) -> Node:
    """Find root token and build tree from it"""
    if is_semantic:
        root_tokens = [token for token in sentence if token.sem_dep.head_idx == -1]
    else:
        root_tokens = [token for token in sentence if token.synt_dep.head_idx == -1]

    if len(root_tokens) > 1:
        # Find root token with the highest depth
        root_token = max(
            root_tokens, key=lambda token: get_root_depth(token, is_semantic, sentence)
        )
        # Set other roots as children of root token
        for other_root_token in root_tokens:
            if other_root_token != root_token:
                if is_semantic:
                    other_root_token.sem_dep.head_idx = root_token.idx
                    root_token.sem_dep.child_indices.append(other_root_token.idx)
                else:
                    other_root_token.synt_dep.head_idx = root_token.idx
                    root_token.synt_dep.child_indices.append(other_root_token.idx)
    else:
        root_token = next(iter(root_tokens))

    if is_semantic:
        dep_relation = root_token.sem_dep
    else:
        dep_relation = root_token.synt_dep

    root_node = Node(
        root_token.idx,
        parent=None,
        idx=root_token.idx,
        dep=dep_relation.relation,
        text=root_token.text,
        frame_name=root_token.frame_name,
        roles=str(token_roles(root_token, sentence)),
        children=build_nodes(dep_relation.child_indices, sentence, is_semantic),
    )
    return root_node
