import numpy as np 
import cv2
import sys

class PuzzleSolver():
	def __init__(self):
		self.pieces = list()
	
	def import_pieces(self, path_to_pieces):
		pass

	def get_corners(self, img, visualize=False):
	"""Detects the corners of the puzzle piece.
	
	RETURN
	A list of python lists of the x and y coordinates
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
    return corner_list

	def preprocess_pieces(self):
		"""Detect the edges of the puzzle piece and remove background"""
		pass

if __name__ == '__main__':
	pass