import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class DependencyGNN(nn.Module):
    def __init__(
        self,
        num_dependency_labels: int,
        gnn_hidden_dim: int = 100,
        num_gnn_layers: int = 5,
        num_gnn_heads: int = 3,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(num_dependency_labels, gnn_hidden_dim)
        self.gnn_stack = gnn.Sequential(
            "x, edge_index",
            [
                (
                    gnn.GATv2Conv(
                        gnn_hidden_dim,
                        gnn_hidden_dim,
                        heads=num_gnn_heads,
                        concat=True,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(),
                nn.Dropout(
                    dropout_rate,
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
                                ),
                                "x, edge_index -> x",
                            ),
                            (nn.ReLU(), "x -> x"),
                            (
                                nn.Dropout(
                                    dropout_rate,
                                ),
                                "x -> x",
                            ),
                        ]
                        for _ in range(num_gnn_layers - 1)
                    ),
                    [],
                ),
            ],
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> None:
        x = self.node_embedding(x).squeeze()
        gnn_out = self.gnn_stack(x, edge_index)
        return gnn_out
