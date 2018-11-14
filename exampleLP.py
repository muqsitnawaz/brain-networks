# sudo pip install pulp if you don't have it
from pulp import *

def f(x):
    return x*x

# Create the problem, specifying if it's a min or max
prob = LpProblem("Random example problem", LpMaximize)

# c = LpVariable("c", 0, 10, LpInteger)

x = [LpVariable("x1lkashf", 0, None, LpInteger), LpVariable("x1lkadashf", 0, 10, LpInteger)]

# The first equation should always be the single objective function
prob += x[0] + x[1], "maximize this"

# Add the constraints here
  # NOTE: lower and upper bounds of single variables can be set in their constructors above
    # prob += x2 <= 10 is done in the declaration of x2
    # prob += x1,x2 >= 0 is done in the declaration of x1 and x2
prob += -2*x[1] >= 4
prob += 2*x[0] + x[1] <= 18

# Write the problem out to a file used to read in the problem to their solver
prob.writeLP("test.lp")

LpSolverDefault.msg = 1
status = prob.solve()

print("Status:", LpStatus[status])

# Below is how we retrieve the values for our variables
print("x1:", value(x[0]))
print("x2:", value(x[1]))

