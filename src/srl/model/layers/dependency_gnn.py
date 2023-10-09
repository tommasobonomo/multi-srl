import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class DependencyGNN(nn.Module):
    def __init__(
        self,
        num_dependency_labels: int,
        input_embedding_dim: int,
        gnn_hidden_dim: int = 100,
        edge_embedding_dim: int = 100,
        num_gnn_layers: int = 1,
        num_gnn_heads: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_embedding = nn.Embedding(num_dependency_labels, edge_embedding_dim)
        self.gnn_stack = gnn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    gnn.GATv2Conv(
                        input_embedding_dim,
                        gnn_hidden_dim,
                        heads=num_gnn_heads,
                        concat=True,
                        edge_dim=edge_embedding_dim,
                        dropout=dropout_rate,
                    ),
                    "x, edge_index, edge_attr -> x",
                ),
                (
                    nn.ReLU(),
                    "x -> x",
                ),
                *sum(
                    (
                        [
                            (
                                gnn.GATv2Conv(
                                    gnn_hidden_dim * num_gnn_heads,
                                    gnn_hidden_dim,
                                    heads=num_gnn_heads,
                                    concat=True,
                                    edge_dim=edge_embedding_dim,
                                    dropout=dropout_rate,
                                ),
                                "x, edge_index, edge_attr -> x",
                            ),
                            (
                                nn.ReLU(),
                                "x -> x",
                            ),
                        ]
                        for _ in range(num_gnn_layers - 1)
                    ),
                    [],
                ),
            ],
        )

    def forward(
        self,
        node_embeds: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
    ) -> None:
        edge_embeds = self.edge_embedding(edge_attrs.squeeze(-1))
        gnn_out = self.gnn_stack(node_embeds, edge_index, edge_embeds)
        return gnn_out
