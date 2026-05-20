import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

constraints = [
    x + y == 1,
    x >= 0,
    y >= 0
    ]

objective = cp.Maximize(x)

problem = cp.Problem(objective, constraints)
problem.solve()

print(problem.value)
print(x.value, y.value)