from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from src.train import model_configs


def test_every_config_fits_and_predicts():
    X, y = load_wine(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    for name, estimator, params in model_configs():
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        assert len(preds) == len(y_test), name
        assert "model" in params, name
