import pytest
import sys, os, json, tempfile
sys.path.insert(0, 'src')
from data.dataset import CLINC150Dataset, create_dataloader
from transformers import BertTokenizer

@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained('/mnt/data3/wzc/IntentGPT-llama/models/bert-base-uncased')

@pytest.fixture
def sample_data(tmp_path):
    data = [
        {"text": "check my balance", "intent": "balance", "label": 0},
        {"text": "book a flight", "intent": "book_flight", "label": 1},
        {"text": "what is 2+2?", "intent": "oos", "label": 150},
    ]
    data_file = tmp_path / "test.json"
    with open(data_file, 'w') as f:
        json.dump(data, f)
    return str(data_file)

def test_dataset_loads(sample_data, tokenizer):
    ds = CLINC150Dataset(sample_data, tokenizer)
    assert len(ds) == 3

def test_dataset_oos_ratio(sample_data, tokenizer):
    ds = CLINC150Dataset(sample_data, tokenizer)
    ratio = ds.get_oos_ratio()
    assert abs(ratio - 1/3) < 0.01

def test_sample_format(sample_data, tokenizer):
    ds = CLINC150Dataset(sample_data, tokenizer)
    item = ds[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'labels' in item
    assert 'is_oos' in item

def test_oos_label_correct(sample_data, tokenizer):
    ds = CLINC150Dataset(sample_data, tokenizer)
    oos_item = ds[2]
    assert oos_item['labels'].item() == 150
    assert oos_item['is_oos'].item() == True
