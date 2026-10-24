import pytest
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.dora.models import (
    detect_model_family,
    get_target_linear_module_names,
    setup_adapted_model,
)
from src.dora.layer import AdaptiveRankLinear


class TestMultiModelArchitecture:
    def test_model_family_detection(self):
        assert detect_model_family("roberta-base") == "roberta"
        assert detect_model_family("distilroberta-base") == "roberta"
        assert detect_model_family("bert-base-uncased") == "bert"
        assert detect_model_family("distilbert-base-uncased") == "distilbert"

    def test_roberta_adaptation(self):
        cfg = AutoConfig.from_pretrained("roberta-base", num_labels=2)
        model = AutoModelForSequenceClassification.from_config(cfg)

        model, replaced, param_info = setup_adapted_model(model, initial_rank=3, alpha=2.0)
        assert len(replaced) == 72
        assert param_info["adapted_layer_count"] == 72
        assert param_info["initial_total_rank"] == 216
        assert param_info["trainable_parameters"] > 0
        assert param_info["trainable_parameters"] < param_info["total_parameters"]

        # Forward pass smoke test
        x = torch.randint(0, 1000, (2, 16))
        attn = torch.ones((2, 16), dtype=torch.long)
        out = model(input_ids=x, attention_mask=attn)
        assert out.logits.shape == (2, 2)
        assert not torch.isnan(out.logits).any()

    def test_distilroberta_adaptation(self):
        cfg = AutoConfig.from_pretrained("distilroberta-base", num_labels=2)
        model = AutoModelForSequenceClassification.from_config(cfg)

        model, replaced, param_info = setup_adapted_model(model, initial_rank=3, alpha=2.0)
        assert len(replaced) == 36
        assert param_info["adapted_layer_count"] == 36
        assert param_info["initial_total_rank"] == 108

        x = torch.randint(0, 1000, (2, 16))
        attn = torch.ones((2, 16), dtype=torch.long)
        out = model(input_ids=x, attention_mask=attn)
        assert out.logits.shape == (2, 2)
        assert not torch.isnan(out.logits).any()

    def test_bert_adaptation(self):
        cfg = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2)
        model = AutoModelForSequenceClassification.from_config(cfg)

        model, replaced, param_info = setup_adapted_model(model, initial_rank=3, alpha=2.0)
        assert len(replaced) == 72
        assert param_info["adapted_layer_count"] == 72
        assert param_info["initial_total_rank"] == 216

        x = torch.randint(0, 1000, (2, 16))
        attn = torch.ones((2, 16), dtype=torch.long)
        out = model(input_ids=x, attention_mask=attn)
        assert out.logits.shape == (2, 2)
        assert not torch.isnan(out.logits).any()

    def test_distilbert_adaptation(self):
        cfg = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=2)
        model = AutoModelForSequenceClassification.from_config(cfg)

        model, replaced, param_info = setup_adapted_model(model, initial_rank=3, alpha=2.0)
        assert len(replaced) == 36
        assert param_info["adapted_layer_count"] == 36
        assert param_info["initial_total_rank"] == 108

        x = torch.randint(0, 1000, (2, 16))
        attn = torch.ones((2, 16), dtype=torch.long)
        out = model(input_ids=x, attention_mask=attn)
        assert out.logits.shape == (2, 2)
        assert not torch.isnan(out.logits).any()

    def test_classifier_head_trainability_switch(self):
        cfg = AutoConfig.from_pretrained("roberta-base", num_labels=2)
        model = AutoModelForSequenceClassification.from_config(cfg)

        _, _, info_frozen = setup_adapted_model(model, initial_rank=3, train_classifier_head=False)
        assert info_frozen["classifier_head_parameters"] == 0

        model2 = AutoModelForSequenceClassification.from_config(cfg)
        _, _, info_trainable = setup_adapted_model(model2, initial_rank=3, train_classifier_head=True)
        assert info_trainable["classifier_head_parameters"] > 0
        assert info_trainable["trainable_parameters"] > info_frozen["trainable_parameters"]
