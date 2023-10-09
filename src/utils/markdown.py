from src.token_types import Sentence, TokenFeatures


def sanitize_mermaid_text(text: str | None) -> str:
    if text is None:
        return ""
    elif '"' in text:
        return text.replace('"', "#quot;")
    else:
        return f'"{text}"'


def sentence_to_token_index_dict(sentence: Sentence) -> dict[int, TokenFeatures]:
    """
    Maps a sentence, i.e. a list of TokenFeatures, to a dict with the index of each token as key and the token as value.
    This is completely overkill, as indexing the list or the dict would usually give the same result,
    but it may happen that tokens are permutated and not in the exact order in the sentence,
    so it makes the indexing slightly more robust.
    """
    return {token.idx: token for token in sentence}


def to_markdown_graph(token_features: dict[int, TokenFeatures]) -> str:
    """Function that represents a sentence represented with token features as a Markdown Mermaid graph.
    More specifically, it renders the DEP-tree of the sentence left-to-right."""
    mermaid_graph = "```mermaid\n"
    mermaid_graph += "graph TD\n"

    # Find root token
    root_token = next(
        filter(lambda token: token.dep == "ROOT", token_features.values())
    )

    # Start BFS on root node
    node_queue = [root_token.idx]
    for node_idx in node_queue:
        # Find node, add its children to queue
        node = token_features[node_idx]
        node_text = sanitize_mermaid_text(node.text)
        node_queue += node.dep_child_idx

        # Write node info to graph
        for child_node_idx in node.dep_child_idx:
            child_node = token_features[child_node_idx]
            child_text = sanitize_mermaid_text(child_node.text)
            mermaid_graph += f"{node.idx}[{node_text}] -->|{child_node.dep}| {child_node.idx}[{child_text}]\n"

        # Add styling for tokens
    for token in token_features.values():
        # Frames in red
        if token.frame_name is not None:
            mermaid_graph += f"style {token.idx} fill:#4169e1\n"

        # Roles in orange if semantic head is given role, otherwise yellow and red
        for role in token.frame_roles:
            if role.role_idx == role.semantic_head_idx:
                mermaid_graph += f"style {token.idx} fill:#A04000\n"
            else:
                mermaid_graph += f"style {role.role_idx} fill:#B7950B\n"
                if role.semantic_head_idx is not None:
                    print(role.semantic_head_idx)
                    mermaid_graph += f"style {role.semantic_head_idx} fill:#AB2328\n"

    mermaid_graph += "```\n\n"
    return mermaid_graph
