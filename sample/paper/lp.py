from pulp import *

prob = LpProblem("problem1", LpMaximize)
x1 = LpVariable("x1", 0, None, LpContinuous)
x2 = LpVariable("x2", 0, None, LpContinuous)
x3 = LpVariable("x3", 0, None, LpContinuous)
prob += 1000 * x1 + 2000 * x2 + 3000 * x3
prob += x1 + 2 * x2 + 3 * x3 <= 10
prob += 0 * x1 + x2 + 2 * x3 <= 5
prob.writeLP("problem1.lp")
prob.solve()
print("status:", LpStatus[prob.status])
for v in prob.variables():
    print(v.name, v.varValue)