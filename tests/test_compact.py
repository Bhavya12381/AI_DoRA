import pytest
import torch
from torch import nn

from src.dora.layer import AdaptiveRankLinear
from src.dora.compact import (
    CompactAdaptiveRankLinear,
    export_compact_layer,
    export_compact_model,
    verify_compact_equivalence,
    measure_model_parameter_stats,
)


class TestCompactAdapter:
    def test_all_components_retained(self):
        base = nn.Linear(16, 32)
        layer = AdaptiveRankLinear(base, rank=4, alpha=2.0)
        layer.c.data.fill_(1.0)

        compact = export_compact_layer(layer)
        assert compact.active_rank == 4
        assert compact.original_rank == 4
        assert compact.scale == layer.scale

        x = torch.randn(4, 16)
        y_gated = layer(x)
        y_compact = compact(x)
        assert torch.allclose(y_gated, y_compact, atol=1e-6)

    def test_partial_components_pruned(self):
        base = nn.Linear(16, 32)
        layer = AdaptiveRankLinear(base, rank=4, alpha=2.0)
        # Prune components 0 and 2
        layer.c.data.copy_(torch.tensor([0.0, 1.5, 0.0, -0.8]))

        compact = export_compact_layer(layer)
        assert compact.active_rank == 2
        assert compact.original_rank == 4
        assert compact.A.shape == (2, 16)
        assert compact.B.shape == (32, 2)
        assert compact.c.shape == (2,)

        x = torch.randn(8, 16)
        y_gated = layer(x)
        y_compact = compact(x)
        max_diff = (y_gated - y_compact).abs().max().item()
        assert max_diff < 1e-6

    def test_all_components_pruned_zero_rank(self):
        base = nn.Linear(16, 32)
        layer = AdaptiveRankLinear(base, rank=3, alpha=2.0)
        layer.c.data.zero_()

        compact = export_compact_layer(layer)
        assert compact.active_rank == 0
        assert compact.A is None
        assert compact.B is None
        assert compact.c is None
        assert compact.active_parameter_count() == 0

        x = torch.randn(4, 16)
        y_gated = layer(x)
        y_compact = compact(x)
        y_base = base(x)

        assert torch.allclose(y_gated, y_compact, atol=1e-7)
        assert torch.allclose(y_compact, y_base, atol=1e-7)

    def test_model_compaction_and_equivalence(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = AdaptiveRankLinear(nn.Linear(8, 16), rank=4, alpha=2.0)
                self.fc2 = AdaptiveRankLinear(nn.Linear(16, 2), rank=4, alpha=2.0)

            def forward(self, input_ids):
                return self.fc2(torch.relu(self.fc1(input_ids)))

        net = SimpleNet()
        # fc1 retains 2 components, fc2 retains 3 components
        net.fc1.c.data.copy_(torch.tensor([1.0, 0.0, 2.0, 0.0]))
        net.fc2.c.data.copy_(torch.tensor([0.0, 1.2, 0.5, 1.1]))

        compact_net, summary = export_compact_model(net)

        assert summary["adapted_layer_count"] == 2
        assert summary["adapter_parameter_reduction"] > 0
        assert summary["adapter_parameter_reduction_percent"] > 0

        sample_batch = {"input_ids": torch.randn(5, 8)}
        is_equiv, max_diff = verify_compact_equivalence(net, compact_net, sample_batch)
        assert is_equiv
        assert max_diff < 1e-6

    def test_parameter_measurement_stats(self):
        base = nn.Linear(10, 20)
        layer = AdaptiveRankLinear(base, rank=4, alpha=1.0)
        layer.c.data.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0]))

        model = nn.Sequential(layer)
        stats_gated = measure_model_parameter_stats(model)
        assert stats_gated["total_possible_components"] == 4
        assert stats_gated["adapter_active_components"] == 2

        compact_model, _ = export_compact_model(model)
        stats_compact = measure_model_parameter_stats(compact_model)
        assert stats_compact["adapter_allocated_parameters"] < stats_gated["adapter_allocated_parameters"]
