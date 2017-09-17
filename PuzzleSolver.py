import numpy as np 
import cv2
import sys

class PuzzleSolver():
	def __init__(self):
		self.pieces = list()
	
	def import_pieces(self, path_to_pieces):
		pass

	def get_corners(img, visualize=False):
	"""Detects the corners of the puzzle piece.
	
	RETURN
	A list of python lists of the x and y coordinates beginning with the top-left and going clockwise
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	gray = cv2.GaussianBlur(gray,(3,3),0)
	corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 10)
	corners = np.int0(corners)
	
	# Clean up output
	corner_list = list()
	for i in corners:
		x, y = i.ravel()
		corner_list.append([x, y])
		if visualize:
			cv2.circle(img, (x, y), 3, 255, -1)
	if visualize:
		cv2.imshow("image", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# Order the corners
	top_corners = list()
	# Split into top and bottom
	for i in range(2):
		top = min(corner_list, key=lambda x: x[1])
		corner_list.remove(top)
		top_corners.append(top)
	bottom_corners = list(corner_list)
	# Split into left and right
	tl = min(top_corners, key=lambda x: x[0])
	top_corners.remove(tl)
	tr = top_corners[0]
	bl = min(bottom_corners, key=lambda x: x[0])
	bottom_corners.remove(bl)
	br = bottom_corners[0]

	return tl, tr, br, bl

	def preprocess_pieces(self):
		"""Detect the edges of the puzzle piece and remove background"""
		pass

if __name__ == '__main__':
	pass