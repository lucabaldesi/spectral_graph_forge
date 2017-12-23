#!/usr/bin/env python
# http://stackoverflow.com/questions/9809436/fast-solution-to-subset-sum/9931260


from bisect import bisect
import numpy as nm


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


def left_closest(D, v):
    '''Return the key left-closest to v'''
    values = sorted(D.keys())
    i = bisect(values, v)

    if i >= len(values) or (i > 0 and
                            abs(values[i] - v) > abs(values[i-1] - v)):
        return values[i-1]

    return values[i]


def ssum(X, c, s):
    """ Simple impl. of the polytime approximate subset sum algorithm
    Returns True if the subset exists within our given error; False otherwise
    """
    S = [0]
    N = len(X)
    for xi in X:
        T = [xi + y for y in S]
        U = set().union(T, S)
        U = sorted(U)  # Coercion to list
        S = []
        y = U[0]
        S.append(y)
        for z in U:
            if y + (c*s)/N < z and z <= s:
                y = z
                S.append(z)
    if not c:  # For zero error, check equivalence
        return S[bisect(S, s)-1] == s
    return bisect(S, (1 - c) * s) != bisect(S, s)


def ssum_elements(X, c, s):
    """ Simple impl. of the polytime approximate subset sum algorithm
    Returns the list of elements
    """
    S = {0: []}
    N = len(X)
    for xi in X:
        T = {(xi + y): S[y] + [xi] for y in S}
        U = merge_two_dicts(S, T)
        # U = sorted(U)  # Coercion to list
        y = min(U)
        S = {y: U[y]}
        for z in U:
            if y + (c*s)/N < z and z <= s:
                y = z
                S[z] = U[z]
    if not c:  # For zero error, check equivalence
        if s in S:
            return S[s]
        else:
            return []
    else:
        left_closest = S.keys()[bisect(S.keys(), s)-1]
        if left_closest > (1 - c) * s:
            return S[left_closest]
        else:
            return []


def ssum_index(X, c, s):
    """ Simple impl. of the polytime approximate subset sum algorithm
    Returns the list of index for the row vector X

    Keyword arguments:
    X -- non-empty numpy column vector
    c -- accepted precision
    s -- target sum
    """
    S = {0: []}
    N = X.size
    for i in range(N):
        T = {(X[i, 0] + y): S[y] + [i] for y in S}
        U = merge_two_dicts(S, T)
        y = min(U)
        S = {y: U[y]}
        for z in sorted(U.iterkeys()):
            # if y + (c*s)/float(N) < z and z <= s:
            #     y = z
            if z <= s + c:
                S[z] = U[z]
    if not c:  # For zero error, check equivalence
        if s in S:
            return S[s]
        else:
            return []
    else:
        candidate = left_closest(S, s)
        if abs(candidate - s) < c:
            return S[candidate]
        else:
            return []


def sssum(X, C, eps):
    '''
    Subset sum problem solver executing in pseudo polynomial time.
    Returns the index of the selected items from X

    Keyword arguments:
    X -- non-empty numpy vector
    C -- target sum
    eps -- accepted error
    '''
    x = nm.ravel(X.copy())
    n = len(x)
    m = min(x)

    if not n or C < m:
        return []

    M = ssum_table(x, C, eps, m)

    # Backtracking the solution
    S = ssum_solution(M, C, x, eps)

    if S:
        return S
    return []


def posnegsort(x):
    n = len(x)
    i = 0
    j = n-1
    indexes = nm.ravel(nm.zeros((1, n)))
    while i < j:
        while i < n and x[i] >= 0 and i != j:
            indexes[i] = i
            i += 1
        while j >= 0 and x[j] < 0 and i != j:
            indexes[j] = j
            j -= 1
        if i < j:
            tmp = x[i]
            x[i] = x[j]
            x[j] = tmp
            indexes[j] = i
            indexes[i] = j
            i += 1
            j -= 1
            if i == j:
                indexes[i] = i
        elif i < n:
            indexes[i] = i
    while i < n and x[i] >= 0:
        i += 1
    return x, i, indexes


def ssum_table(x, target, eps, m=None):
    n = len(x)
    if not n:
        return nm.zeros((1, 1))
    if m is None:
        m = min(x)
    c = nm.linspace(m, target + eps, round((target + eps - m) / eps) + 1)
    D = len(c)
    M = nm.zeros((n, D))

    for d in range(D):
            if x[0] > c[d]:
                M[0, d] = 0
            else:
                M[0, d] = x[0]

    for i in range(1, n):
        for d in range(D):
            if x[i] > c[d]:
                M[i, d] = M[i - 1, d]
            else:
                if c[d] - x[i] >= m:
                    j = d - int(round(x[i] / eps))
                    M[i, d] = max(M[i - 1, d], M[i - 1, j] + x[i])
                else:
                    M[i, d] = max(M[i - 1, d], x[i])
    return M


def best_memory_position(M, C, eps, start=0):
    n, D = M.shape
    a = start
    b = D
    d = (a + b) / 2
    v = M[n-1, d] - C
    while abs(v) > eps and a < b - 1:
        if v > 0:
            b = d
        else:
            a = d
        # print "n: " + str(n) + " pos: " + str(d) + " value: " + str(v)
        d = (a + b) / 2
        v = M[n-1, d] - C

    return d


def ssum_solution(M, C, x, eps):
    S = []
    n, D = M.shape
    if n == 0 or D == 0 or len(x) == 0:
        return S
    d = D - 1
    i = n - 1
    d = best_memory_position(M, C, eps)
    # t = M[i, d]
    # if abs(M[i, d] - C) <= eps:
    while i >= 0:
        if i > 0 and eps * d - x[i] + eps > 0:
            # not first element and its value is not equal or greater than\
            #    the residual
            j = d - int(round(x[i] / eps))
            if M[i, d] == M[i - 1, j] + x[i]:
                S.append(i)
                d = j
        else:
            # First element or element greater or equal than the target sum
            if M[i, d] == x[i]:
                S.append(i)
                i = -1
        i -= 1

    # assert(abs(t - sum(x[S])) <= eps)
    return S


def sdiff_solution(Mp, Mn, C, x, eps, k=0):
    np, Dp = Mp.shape
    nn, Dn = Mn.shape

    j = 0
    i = best_memory_position(Mp, C, eps/2)  # check for positive values solution
    v = Mp[np - 1, i] - C
    best = v
    S = ssum_solution(Mp, Mp[np - 1, i], x[:np], eps/2)
    n = len(S)
    ind_sum = -1

    while (abs(v) > eps or (k != 0 and n-k != 0)) and ind_sum != i + j:
        ind_sum = i + j
        if v > 0 and j < Dn - 1:
            j = best_memory_position(Mn, Mp[np - 1, i] - C, eps/2, j+1)
        elif v < 0 and i < Dp - 1:
            i = best_memory_position(Mp, Mn[nn - 1, j] + C, eps/2, i+1)
        elif i < Dp - 1 and j < Dn - 1:
            i += 1
            j += 1
        v = Mp[np - 1, i] - Mn[nn - 1, j] - C
        if abs(v) - abs(best) <= eps:
            R = ssum_solution(Mp, Mp[np - 1, i], x[:np], eps/2) +\
                [np + s for s in
                 ssum_solution(Mn, Mn[nn - 1, j], x[np:], eps/2)]
            new_n = len(R)
            if ((k == 0 or new_n == n) and abs(best) > abs(v)) or \
               (k > 0 and abs(new_n-k) < abs(n-k)):
                S = R
                n = new_n
                # print "Value: " + str(v) + " i: " + str(i) + "/" + str(Dp) +\
                #     " j: " + str(j) + "/" + str(Dn) +\
                #     " n: " + str(n) + "/" + str(k)
                best = v
    if abs(C) + eps < abs(best):
        S = []
    return S


def gssum(X, C, eps, lenght=0):
    '''
    Generalized subset sum problem solver executing in pseudo polynomial time.
    Returns the index of the selected items from X

    Keyword arguments:
    X -- non-empty numpy vector
    C -- target sum
    eps -- accepted error
    '''
    # print "GSSum: X:" + str(nm.ravel(X)) + " C:" + str(C) + " eps:" + \
    #    str(eps) + " " + str(lenght)
    x = nm.ravel(X.copy())
    n = len(x)
    eps = float(eps)
    if not n:
        return []
    if C < 0:
        x = x * -1
        C = C * -1

    x, k, ind = posnegsort(x)
    x[k:] = x[k:] * -1

    Sp = sum(x[:k])
    Sn = sum(x[k:])
    if Sp >= Sn:
        Cp = Sn + C + eps
        Cn = Sn
    else:
        Cp = Sp
        Cn = Sp - C
        if Cn < min(x[k:]):
            Cn = min(x[k:])

    if k < 1 or C + eps < min(x[:k]) - Sn:
        return []
    Mp = ssum_table(x[:k], Cp, eps/2)
    Mn = ssum_table(x[k:], Cn, eps/2)
    S = sdiff_solution(Mp, Mn, C, x, eps, lenght)

    if S:
        return [ind[s] for s in S]
    return []
