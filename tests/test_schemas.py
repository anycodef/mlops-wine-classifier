from api.schemas import CANONICAL_COLUMNS, WineSample


def test_to_row_maps_to_canonical_columns():
    sample = WineSample(
        alcohol=14.23, malic_acid=1.71, ash=2.43, alcalinity_of_ash=15.6,
        magnesium=127, total_phenols=2.8, flavanoids=3.06,
        nonflavanoid_phenols=0.28, proanthocyanins=2.29, color_intensity=5.64,
        hue=1.04, od280_od315_of_diluted_wines=3.92, proline=1065,
    )
    row = sample.to_row()
    assert list(row.keys()) == CANONICAL_COLUMNS
    assert row["od280/od315_of_diluted_wines"] == 3.92
    assert len(row) == 13
