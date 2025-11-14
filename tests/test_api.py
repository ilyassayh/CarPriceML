import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app, META_PATH


client = TestClient(app)


def test_health():
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert 'status' in data


@pytest.mark.skipif(not Path(META_PATH).exists(), reason='Model metadata missing; train the model first')
def test_predict_minimal():
    # Load expected columns and post a minimal payload (all None)
    meta = json.loads(Path(META_PATH).read_text(encoding='utf-8'))
    expected_cols = meta.get('numeric_features', []) + meta.get('categorical_features', [])
    payload = {k: None for k in expected_cols}
    resp = client.post('/predict', json=payload)
    assert resp.status_code == 200
    assert 'price' in resp.json()


