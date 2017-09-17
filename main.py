from PuzzleSolver import *

if __name__ == '__main__':
	solver = PuzzleSolver()
	solver.import_pieces('../Puzzle2')
	print(solver.solve())
	print(solver.convex_edges)
	print(solver.concave_edges)
	print(solver.straight_edges)