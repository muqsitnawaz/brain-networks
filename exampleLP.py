# sudo pip install pulp if you don't have it
from pulp import *

# Create the problem, specifying if it's a min or max
prob = LpProblem("Random example problem", LpMaximize)

# Define the variables
x1 = LpVariable("x1", 0, None, LpInteger)
x2 = LpVariable("x2", 0, 10, LpInteger)

# The first equation should always be the single objective function
prob += 2*x1 + 3*x2, "maximize this"

# Add the constraints here
  # NOTE: lower and upper bounds of single variables can be set in their constructors above
    # prob += x2 <= 10 is done in the declaration of x2
    # prob += x1,x2 >= 0 is done in the declaration of x1 and x2
prob += x1 - 2*x2 <= 4
prob += 2*x1 + x2 <= 18

# Write the problem out to a file used to read in the problem to their solver
prob.writeLP("test.lp")

status = prob.solve()

print("Status:", LpStatus[status])

# Below is how we retrieve the values for our variables
print("x1:", value(x1))
print("x2:", value(x2))

