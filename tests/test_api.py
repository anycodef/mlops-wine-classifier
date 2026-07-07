from fastapi.testclient import TestClient

from api import main
from api.schemas import WineSample


class _StubModel:
    def predict(self, frame):
        return [0] * len(frame)


SAMPLE = dict(
    alcohol=14.23, malic_acid=1.71, ash=2.43, alcalinity_of_ash=15.6,
    magnesium=127, total_phenols=2.8, flavanoids=3.06,
    nonflavanoid_phenols=0.28, proanthocyanins=2.29, color_intensity=5.64,
    hue=1.04, od280_od315_of_diluted_wines=3.92, proline=1065,
)


def test_health():
    with TestClient(main.app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_with_stub_model():
    with TestClient(main.app) as client:
        main.state["model"] = _StubModel()
        resp = client.post("/predict", json={"samples": [SAMPLE, SAMPLE]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictions"] == [0, 0]
    assert body["model_alias"] == "champion"


def test_winesample_validates():
    assert WineSample(**SAMPLE).proline == 1065
