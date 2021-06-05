import numpy as np
from infinite_selection import InfFS


def test_fit():
    X = np.array([[1, 2, 5], [-3, -4, 8], [5, 6, 1], [7, 8, 1]])
    y = np.array([0, 1, 0, 1])

    inf = InfFS()
    [RANKED, WEIGHT] = inf.infFS(X, y, alpha=0.5, supervision=1, verbose=1)

    assert np.shape(RANKED) == (3,)
    assert np.shape(WEIGHT) == (3,)
    assert (np.array(WEIGHT) >= 0).all()
