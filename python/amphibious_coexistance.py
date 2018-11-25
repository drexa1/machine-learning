from ortools.linear_solver import pywraplp

def solve_coexistence():
    name = 'Amphibian coexistence'
    solver = pywraplp.Solver(name, pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    x = [solver.NumVar(0, 1000,'x[%i]' % i) for i in range(3)]
    obj = solver.NumVar(0, 3000, 'obj')
    solver.Add(2*x[0] + x[1] + x[2] <= 1500)
    solver.Add(x[0] + 3*x[1] + 2*x[2] <= 3000)
    solver.Add(x[0] + 2*x[1] + 3*x[2] <= 4000)
    solver.Add(obj == x[0] + x[1] + x[2])
    solver.Maximize(obj)
    solver.Solve()
    return obj.SolutionValue(),[e.SolutionValue() for e in x]

obj, x = solve_coexistence()
T = [['Specie', 'Count']]
for i in range(3):
    T.append([['Toads','Salamanders','Caecilians'][i], x[i]])
T.append(['Total', obj])
for e in T:
    print (e[0], e[1])
