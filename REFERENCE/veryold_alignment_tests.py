# Some definitions
lam = 1e8     # lambda (Lagrange multiplier for coordinate system)
Na = 12       # number of alignables A_i
fixed = None  # which alignable to be held fixed (None for average of all)

# A representation of the measurements m_ij
class Measurement:
    def __init__(self, i, j, value, uncertainty):
        self.i, self.j, self.value, self.uncertainty = i, j, value, uncertainty

# measurements = [
#     Measurement(0, 1,  0.12, 0.01),
#     Measurement(1, 2,  0.00, 0.01),
#     Measurement(2, 3, -0.15, 0.01),
#     Measurement(3, 4, -0.15, 0.01),
#     Measurement(4, 0,  0.18, 0.01),
#     ]
measurements = [
    Measurement( 0,  1,  0.1, 0.01),
    Measurement( 1,  2, -0.2, 0.01),
    Measurement( 2,  3, -0.2, 0.01),
    Measurement( 3,  0,  0.3, 0.01),
    Measurement( 0,  6,  0.2, 0.01),
    Measurement( 6,  7,  0.1, 0.01),
    Measurement( 7,  1, -0.4, 0.01),
    Measurement( 7,  8,  0.2, 0.01),
    Measurement( 8,  1,  0.2, 0.01),
    Measurement( 8,  9, -0.3, 0.01),
    Measurement( 9,  2,  0.3, 0.01),
    Measurement( 9, 10, -0.1, 0.01),
    Measurement(10,  2, -0.2, 0.01),
    Measurement(10, 11, -0.1, 0.01),
    Measurement(11,  3,  0.5, 0.01),
    Measurement(11,  4, -0.4, 0.01),
    Measurement( 4,  3, -0.1, 0.01),
    Measurement( 4,  5,  0.1, 0.01),
    Measurement( 5,  0, -0.3, 0.01),
    Measurement( 5,  6,  0.1, 0.01),
    ]
Nm = len(measurements)   # number of measurements m_ij

# objective function (for an arbitrary number of alignables)
def chi2_arbitrary(*args):
    if fixed is None: s = lam * sum(args)**2
    else: s = lam * args[fixed]**2
    for m in measurements:
        s += (m.value - args[m.i] + args[m.j])**2 / m.uncertainty**2
    return s

# objective function for Na alignables (in a form that Minuit can accept)
chi2_arguments = ", ".join(["A%i" % i for i in range(Na)])
chi2 = eval("lambda %s: chi2_arbitrary(%s)" % (chi2_arguments, chi2_arguments))

# minimize the objective function using Minuit
import minuit
minimizer = minuit.Minuit(chi2)
minimizer.migrad()
print [minimizer.values["A%i" % i] for i in range(Na)]

# minimize the objective function using linear algebra
from numpy import matrix
from numpy.linalg.linalg import inv, dot

# Equation 15
def vk(k):
    s = 0.
    for m in measurements:
        d = 1.*m.value/m.uncertainty**2
        if m.i == k: s += d
        if m.j == k: s -= d
    return s

# Equation 16
def Akl(k, l):
    if fixed is None: s = lam
    else:
        if k == 0 and l == 0: s = lam
        else: s = 0.
    for m in measurements:
        d = 1./m.uncertainty**2
        if k == l and (m.i == k or m.j == k): s += d
        if (m.i == k and m.j == l) or (m.j == k and m.i == l): s -= d
    return s

# Equation 17
v = [vk(k) for k in range(Na)]
M = matrix([[Akl(k, l) for k in range(Na)] for l in range(Na)])
print dot(inv(M), v)

def print_matrix(mat):
    if not isinstance(mat, dict):
        m = {}
        for i in range(Na):
            for j in range(Na):
                m["A%d" % i, "A%d" % j] = mat[i, j]
    else:
        m = mat
    print "\n".join([" ".join(["%8.2g" % m["A%d" % i, "A%d" % j]
                               for j in range(Na)]) for i in range(Na)])

print_matrix(minimizer.covariance)
print_matrix(inv(M))

########################
















##############################################################

from numpy import matrix
from numpy.linalg.linalg import inv, dot, eig, diagonal
    
class CSCPairConstraint:
    def __init__(self, i, j, value, error):
        self.i, self.j, self.value, self.error = i, j, value, error
        if self.i == self.j: raise Exception

    def chi2(self, A):
        return (self.value - A[self.i] + A[self.j])**2 / self.error**2

    def deriv1(self, A, k):
        d = (2./self.error**2) * (self.value - A[self.i] + A[self.j])
        if k == self.i:   d *= -1.
        elif k == self.j: d *= 1.
        else:             d *= 0.
        return d
    
    def deriv2(self, A, k, l):
        d = (2./self.error**2)
        if k == self.i:   d *= -1.
        elif k == self.j: d *= 1.
        else:             d *= 0.
        if l == self.i:   d *= -1.
        elif l == self.j: d *= 1.
        else:             d *= 0.
        return d

def wmean(xlist):
  s, w = 0., 0.
  for x, e in xlist:
    if e > 0.:
      wi = 1./e**2
      s += x*wi
      w += wi
  return s/w, sqrt(1./w)

def v(k):
    s = 0.
    for constraint in constraints:
        d = 2.*constraint.value/constraint.error**2
        if constraint.i == k: s += d
        if constraint.j == k: s -= d
    return s

### fix the average of 5-N
def M(k, l):
    if k > 3 and l > 3:
        s = 1e-6*2.
    else:
        s = 0.
    for constraint in constraints:
        d = 2./constraint.error**2
        if k == l and (constraint.i == k or constraint.j == k): s += d
        if (constraint.i == k and constraint.j == l) or (constraint.j == k and constraint.i == l): s -= d
    return s
def chi2(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21):
    A = [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21]
    s = 1e-6*(A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + A12 + A13 + A14 + A15 + A16 + A17 + A18 + A19 + A20 + A21)**2
    for constraint in constraints:
        s += constraint.chi2(A)
    return s

N = 4 + 18
constraints = [
    CSCPairConstraint(0, 4 +  0, 0., 1.),
    CSCPairConstraint(0, 4 +  1, 0., 1.),
    CSCPairConstraint(0, 4 +  2, 0., 1.),
    CSCPairConstraint(0, 4 +  3, 0., 1.),
    CSCPairConstraint(0, 4 +  4, 0., 1.),
    CSCPairConstraint(0, 4 +  5, 0., 1.),
    CSCPairConstraint(0, 4 +  6, 0., 1.),
    CSCPairConstraint(0, 4 +  7, 0., 1.),
    CSCPairConstraint(0, 4 +  8, 0., 1.),
    CSCPairConstraint(0, 4 +  9, 0., 1.),
    CSCPairConstraint(0, 4 + 10, 0., 1.),
    CSCPairConstraint(0, 4 + 11, 0., 1.),
    CSCPairConstraint(0, 4 + 12, 0., 1.),
    CSCPairConstraint(0, 4 + 13, 1., 1.),
    CSCPairConstraint(0, 4 + 14, 0., 1.),
    CSCPairConstraint(0, 4 + 15, 0., 1.),
    CSCPairConstraint(0, 4 + 16, 0., 1.),
    CSCPairConstraint(0, 4 + 17, 0., 1.),

    CSCPairConstraint(1, 4 +  1, 0., 1.),
    CSCPairConstraint(1, 4 +  9, 0., 1.),
    CSCPairConstraint(2, 4 +  3, 0., 1.),
    CSCPairConstraint(2, 4 + 13, 1., 1.),
    CSCPairConstraint(3, 4 +  7, 0., 1.),
    CSCPairConstraint(3, 4 + 15, 0., 1.),

    CSCPairConstraint(4 +  0, 4 +  1, 0., 1.),
    CSCPairConstraint(4 +  1, 4 +  2, 0., 1.),
    CSCPairConstraint(4 +  2, 4 +  3, 0., 1.),
    CSCPairConstraint(4 +  3, 4 +  4, 0., 1.),
    CSCPairConstraint(4 +  4, 4 +  5, 0., 1.),
    CSCPairConstraint(4 +  5, 4 +  6, 0., 1.),
    CSCPairConstraint(4 +  6, 4 +  7, 0., 1.),
    CSCPairConstraint(4 +  7, 4 +  8, 0., 1.),
    CSCPairConstraint(4 +  8, 4 +  9, 0., 1.),
    CSCPairConstraint(4 +  9, 4 + 10, 0., 1.),
    CSCPairConstraint(4 + 10, 4 + 11, 0., 1.),
    CSCPairConstraint(4 + 11, 4 + 12, 0., 1.),
    CSCPairConstraint(4 + 12, 4 + 13, 1., 1.),
    CSCPairConstraint(4 + 13, 4 + 14, -1., 1.),
    CSCPairConstraint(4 + 14, 4 + 15, 0., 1.),
    CSCPairConstraint(4 + 15, 4 + 16, 0., 1.),
    CSCPairConstraint(4 + 16, 4 + 17, 0., 1.),
    CSCPairConstraint(4 + 17, 4 +  0, 0., 1.),
    ]

m = minuit.Minuit(chi2)
m.migrad()
print map(lambda x: "%.2g" % m.values[x], ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21"))

tmp = inv(matrix([[m.covariance["A%d" % i, "A%d" % j]/2. for j in range(N)] for i in range(N)]))
for k in range(N):
    for l in range(N):
        if abs(tmp[k, l]) > 1e-8:
            print "%.2g " % tmp[k, l],
        else:
            print "0 ",
    print

print map(lambda x: "%.2g" % x, [dot(inv(matrix([[M(k, l) for k in range(N)] for l in range(N)])), [v(k) for k in range(N)])[0, i] for i in range(N)])

for k in range(N):
    for l in range(N):
        print "%.2g " % M(k, l),
    print

Minv = inv(matrix([[M(k, l) for k in range(N)] for l in range(N)]))
basis = eig(Minv)[1]
invbasis = inv(basis)
diagonalized = dot(invbasis, dot(Minv, basis))
for i in range(N):
    expression = []
    for j in range(N):
        if abs(invbasis[i, j]) > 1e-9:
            expression.append("%.2g A_{%d}" % (invbasis[i, j], j))
    print " + ".join(expression) + ": %g" % sqrt(diagonal(diagonalized)[i])
