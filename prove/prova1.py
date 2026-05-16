import cvxpy as cp

n = 3

Gamma = cp.Variable((n,n), symmetric=True)

constraints = [
    Gamma >> 0,
    Gamma[0,0] == 1
]

objective = cp.Maximize(Gamma[0,1])

problem = cp.Problem(objective, constraints)

problem.solve()

print(Gamma.value)