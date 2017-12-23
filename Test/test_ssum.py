import numpy as nm
from ssum import ssum_index, sssum, posnegsort, gssum


def test_ssum_index_int():
    x = nm.matrix("7; 9; 9; 2; 4")
    target = 6

    assert(ssum_index(x, 1, target) == [3, 4])
    assert(ssum_index(x, 0.1, target) == [3, 4])
    assert(ssum_index(x, 0, target) == [3, 4])


def test_ssum_index_float():
    x = nm.matrix("0.084503; 0.160734; 0.221232; 0.260074; 0.136729; 0.136729")

    target = 1.9021*0.084503
    assert(ssum_index(x, 0.01, target) == [1])

    target = 1.9021*0.160734
    assert(ssum_index(x, 0.01, target) == [0, 2])

    target = 1.9021*0.221232
    assert(ssum_index(x, 0.01, target) == [1, 3])

    target = 1.9021*0.260074
    assert(ssum_index(x, 0.01, target) == [2, 4, 5])

    target = 1.9021*0.13672
    assert(ssum_index(x, 0.01, target) == [3])


def test_sssum():
    x = nm.matrix("1; 3")
    target = 4
    assert(sssum(x, target, 0.25) == [1, 0])

    x = nm.matrix("7; 9; 9; 2; 4")
    target = 6

    assert(sssum(x, target, 1) == [4, 3])
    assert(sssum(x, target, 0.1) == [4, 3])
    x = nm.matrix("0.084503; 0.160734; 0.221232; 0.260074; 0.136729; 0.136729")

    target = 1.9021*0.084503
    assert(sssum(x, target, 0.01) == [1])

    target = 1.9021*0.160734
    assert(sssum(x, target, 0.001) == [2, 0])

    target = 1.9021*0.221232
    assert(sssum(x, target, 0.01) == [3, 1])

    target = 1.9021*0.260074
    assert(sssum(x, target, 0.01) == [5, 4, 2])

    target = 1.9021*0.13672
    assert(sssum(x, target, 0.01) == [3])

    x = nm.matrix("0.25607264; 0.20030489; 0.08024419; 0.08024419")
    target = 0.45637739162
    assert(sssum(x, target, 0.001) == [1, 0])

    x = nm.matrix("0.00182474  0.04415074  0.08671869  0.19571543  0.08619136\
                  0.45762651 0.1370996   0.33456984  0.62003493  0.03761902")
    target = 0.99404828
    assert(sssum(x, target, 0.005) == [8, 4, 3, 2, 0])

    x = nm.matrix("0.00182474  0.04415074  0.08671869  0.19571543  0.08619136\
                  0.45762651 0.1370996   0.33456984  0.62003493  0.03761902")
    target = 0.99404828
    assert(sssum(x, target, 0.0005) == [9, 8, 7, 0])


def test_posnegsort():
    X = nm.ravel(nm.matrix("1 2 3"))
    x, k, ind = posnegsort(X)
    assert(nm.linalg.norm(x-X) == 0)
    assert(k == 3)
    assert(nm.linalg.norm(ind - nm.ravel(nm.matrix("0 1 2"))) == 0)

    X = nm.ravel(nm.matrix("1 -2 3"))
    x, k, ind = posnegsort(X)
    assert(nm.linalg.norm(x-nm.ravel(nm.matrix("1 3 -2"))) == 0)
    assert(k == 2)
    assert(nm.linalg.norm(ind - nm.ravel(nm.matrix("0 2 1"))) == 0)

    X = nm.ravel(nm.matrix("1 -2 3 -4"))
    x, k, ind = posnegsort(X)
    assert(nm.linalg.norm(x-nm.ravel(nm.matrix("1 3 -2 -4"))) == 0)
    assert(k == 2)
    assert(nm.linalg.norm(ind - nm.ravel(nm.matrix("0 2 1 3"))) == 0)

    X = nm.ravel(nm.matrix("-0.21000653 0.46691924 0.32422917"))
    x, k, ind = posnegsort(X)
    assert(nm.linalg.norm(x-nm.ravel(
        nm.matrix("0.32422917 0.46691924 -0.21000653"))) == 0)
    assert(k == 2)
    assert(nm.linalg.norm(ind - nm.ravel(nm.matrix("2 1 0"))) == 0)


def test_gssum():

    S = gssum(nm.matrix("1 2 3"), 4, 0.5)
    assert(S == [2., 0.])

    S = gssum(nm.matrix("1 -2 3"), 4, 0.5)
    assert(S == [2., 0.])

    S = gssum(nm.matrix("1 -2 -75 3"), 4, 0.5)
    assert(S == [3., 0.])

    S = gssum(nm.matrix("-10 20 -4 6"), 2, 1)
    assert(S == [3., 2.])

    S = gssum(nm.matrix("-2 20 -4 6"), -2, 1)
    assert(S == [0])

    S = gssum(nm.matrix("1 6 8 -2"), 3, 1)
    assert(S == [1., 3.])

    S = gssum(nm.matrix("-0.1 0.2 -0.04 0.06"), 0.02, 0.01)
    assert(S == [3., 2.])

    S = gssum(nm.matrix("0.243653 0.131397 0.319503 0.351513 0.239257 0.091026 \
                        -0.436433 -0.469958 -0.469958"), 0.45090, 0.001)
    assert(S == [2., 1.])

    S = gssum(nm.matrix("0.20030489 0.20030489 0.25607264 0.08024419"), 0.65,
              0.03)
    assert(S == [2., 1., 0])

    S = gssum(nm.matrix("-4 6 -2 -3 5"), 1, 0.5, 3)
    assert(sorted(S) == sorted([3., 2, 1.]))

    S = gssum(nm.matrix("-4 6 -2 -3 5"), 1, 0.5, 2)
    assert(S == [4., 0.])

    S = gssum(nm.matrix("0.20030489 0.20030489 0.08024419 0.08024419"),
              0.416560538684, 0.1, 3)
    assert(S == [3., 1., 0.])

    S = gssum(nm.matrix("0.2560726 0.20030489 0.08024419 0.08024419 0.1828292"),
              0.45637739162, 0.02, 2)
    assert(S == [4., 3., 1.])

    S = gssum(nm.matrix("0.21000686 -0.32422944 0.6953642 -0.46691906 "),
              -0.256912249439, 0.002, 2)
    assert(S == [3, 0])

    S = gssum(nm.matrix("0.21000653 -0.46691924 -0.32422917 "),
              -0.791148944186, 0.001, 3)
    assert(S == [1, 2])

    S = gssum(nm.matrix("0.69536421 -0.32422943"),
              -4.80176618911e-07, 0.001, 2)
    assert(S == [])

    S = gssum(nm.matrix("2 2"),
              4.1, 1, 3)
    assert(S == [1, 0])

    S = gssum(nm.matrix("0.2100065 0.2100065"),
              0.420013347158, 0.001, 3)
    assert(S == [1, 0])

    S = gssum(nm.matrix("-0.46691925  0.21000651"),
              -0.256912132995, 0.001, 2)
    assert(S == [0, 1])

    S = gssum(nm.matrix("0.08619131 -0.05396208 -0.12589051 -0.12589051\
                        -0.04375199 -0.12589051  0.00182487 -0.01268217\
                        0.33456956 -0.04290239 -0.05505203  0.6200346  \
                        -0.14903734 -0.14692704 -0.06159429  0.00182487\
                        -0.07202137 -0.01268217  0.19571567  0.08671863\
                        -0.16048878  0.03761924  0.45762685 -0.12589051\
                        -0.09058065 -0.08943372  0.13709972 -0.14135927\
                        -0.05845963 -0.07202137  0.04415079 -0.12589051\
                        -0.05154028"),
              0.45762647524, 0.0001, 1)
    assert(S == [22])  # also 9,21,24,31

    S = gssum(nm.matrix("-0.05942705 -0.04375211  0.19571543 -0.05845981 \
                        0.08671869 -0.12589039 -0.07202149 -0.07202149 \
                        -0.14903729  0.00182474  0.1370996  -0.12589039 \
                        0.62003493 -0.0615941  -0.05396189 -0.01268229 \
                        -0.04290233 -0.14692716 -0.12589039 -0.08943363 \
                        0.03761902 -0.01268229 -0.14135918  0.33456984  \
                        0.04415074 -0.12589039 -0.05505193 -0.16048883 \
                        -0.09058061 -0.05154019  0.08619136  0.45762651 \
                        -0.12589039"),
              0.490920365029, 0.0001, 4)
    assert(S == [9, 24, 31, 21])

    S = gssum(nm.matrix("-0.05845981 -0.14903729 -0.09058061 -0.05505193\
                        0.08619136  0.45762651 -0.14135918 -0.04375211  \
                        0.62003493 -0.16048883  0.03761902 -0.05396189 \
                        -0.01268229 -0.05154019 -0.12589039  0.33456984 \
                        0.1370996  -0.07202149  0.19571543 -0.14692715 \
                        -0.07202149 -0.12589039  0.08671869 -0.01268229 \
                        -0.04290233  0.04415074  0.00182474 -0.12589039\
                        "),
              1.00128247758, 0.0001, 4)
    assert(S == [8, 15, 16, 2])

    x = nm.matrix("0.03761924  0.08619131 -0.01268217 -0.12589051\
                  -0.12589051 -0.05154028  0.19571566 -0.14692704\
                  -0.07202138  0.04415079 -0.08943372  0.00182487\
                  0.33456956 -0.05845963 -0.01268217  0.00182487\
                  -0.12589051 -0.09058065 -0.05942698 -0.16048878\
                  -0.14135927 -0.14903734  0.13709972  0.08671863\
                  0.62003461 -0.04375199 -0.04290239 -0.07202138")
    target = 0.899553115781
    S = gssum(x, target, 0.0001, 3)
    assert(abs(target - sum(nm.ravel(x)[[int(s) for s in S]])) <= 0.0002)
