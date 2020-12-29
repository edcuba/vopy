import numpy as np
from vopy.pose import triangulate


def test_triangulation():
    P = np.array((
        (-0.538243893729270, -0.996094096614954,	0.176601628617548,	1.81692224917996),
        (0.867232157629573,	-0.523214031686441,	0.755179935677488,	-0.123833350279246),
        (14.8799323173627,	3.51276261416305,	7.04250020114030,	4.44699322469679),
        (1,	1,	1,	1)
    ), dtype=np.float64)
    M1 = np.array(((500., 0., 320., 0.), (0., 500., 240., 0.), (0., 0., 1., 0.)), dtype=np.float64)
    M2 = np.array(((500., 0., 320., -100.), (0., 500., 240., 0.), (0., 0., 1., 0.)), dtype=np.float64)
    p1 = M1.dot(P)
    p2 = M2.dot(P)

    assert p1.shape == p2.shape
    assert p1.shape == (3, 4)

    P_est = triangulate(p1, p2, M1, M2)

    assert P.shape == P_est.shape

    err = (P_est-P)**2
    sum_err = err.sum()

    print(P)
    print(P_est)
    print(P_est - P)
    print(sum_err)

    assert sum_err < 1
