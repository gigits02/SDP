import cvxpy as cp

X = cp.Variable((2, 2), symmetric=True)

constraints = [
    X >> 0,          # X è semidefinita positiva
    X[0, 0] == 1,
    X[1, 1] == 1
]

objective = cp.Maximize(X[0, 1])

problem = cp.Problem(objective, constraints)
problem.solve()

print(problem.value)
print(X.value)