
# Input
# f: a non-linear function
# k: point to approximate at
# Output
# c: c-intercept of the line
# m: slop of the line
def apx(f, k):
    y0 = f(0)
    y1 = f(k)
    slope = (y1 - y0)/(k - 0)
    return (y0, slope)

print(apx(lambda x: x**2 + 3, 5))