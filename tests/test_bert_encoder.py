import pytest
import sys, torch
sys.path.insert(0, 'src')
from models.bert_encoder import BERTIntentEncoder

BERT_MODEL = '/mnt/data3/wzc/IntentGPT-llama/models/bert-base-uncased'

@pytest.fixture(scope='module')
def model():
    return BERTIntentEncoder(BERT_MODEL, num_labels=151)

def test_model_creates(model):
    assert model is not None

def test_forward_shape(model):
    input_ids = torch.randint(0, 1000, (4, 64))
    attention_mask = torch.ones(4, 64, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids, attention_mask)
    assert out['logits'].shape == (4, 151)
    assert out['hidden_states'].shape == (4, 768)

def test_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    assert total > 100_000_000  # BERT-base has ~110M params
