import numpy as np


# -------------------------------------------------
# Question 1: Continuous pair on the unit square
# -------------------------------------------------

def joint_cdf_unit_square(x, y):
    """
    Joint CDF for uniform (X,Y) on unit square.
    """

    if x <= 0 or y <= 0:
        return 0.0

    if 0 < x < 1 and 0 < y < 1:
        return x * y

    if 0 < x < 1 and y >= 1:
        return x

    if x >= 1 and 0 < y < 1:
        return y

    if x >= 1 and y >= 1:
        return 1.0

    return 0.0


def rectangle_probability(x1, x2, y1, y2):
    """
    P(x1 < X <= x2, y1 < Y <= y2)
    using joint CDF formula.
    """

    F = joint_cdf_unit_square

    return (
        F(x2, y2)
        - F(x1, y2)
        - F(x2, y1)
        + F(x1, y1)
    )


def marginal_fx_unit_square(x):
    """
    Marginal PDF of X.
    """

    if 0 < x < 1:
        return 1.0
    return 0.0


def marginal_fy_unit_square(y):
    """
    Marginal PDF of Y.
    """

    if 0 < y < 1:
        return 1.0
    return 0.0


# -------------------------------------------------
# Question 2: Joint PMF, marginals, independence
# -------------------------------------------------

def joint_pmf_heads(x, y):
    """
    Joint PMF from two coin tosses example.
    """

    table = {
        (0, 0): 0.25,
        (0, 1): 0.25,
        (0, 2): 0.0,
        (1, 0): 0.0,
        (1, 1): 0.25,
        (1, 2): 0.25,
    }

    return table.get((x, y), 0.0)


def marginal_px_heads(x):
    """
    P_X(x) = sum over y
    """

    return (
        joint_pmf_heads(x, 0)
        + joint_pmf_heads(x, 1)
        + joint_pmf_heads(x, 2)
    )


def marginal_py_heads(y):
    """
    P_Y(y) = sum over x
    """

    return (
        joint_pmf_heads(0, y)
        + joint_pmf_heads(1, y)
    )


def check_independence_heads():
    """
    Check if X and Y are independent.
    """

    xs = [0, 1]
    ys = [0, 1, 2]

    for x in xs:
        for y in ys:
            pxy = joint_pmf_heads(x, y)
            px = marginal_px_heads(x)
            py = marginal_py_heads(y)

            if not np.isclose(pxy, px * py):
                return False

    return True
