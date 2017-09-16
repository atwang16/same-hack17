import numpy as np 
import cv2
import sys

class PuzzleSolver():
	def __init__(self):
		self.pieces = list()

	def get_edges_test(self):
		"""Takes an image and gives an array same dimension as image that has 1 if
		edge, 0 if not for each pixel."""
		img = cv2.imread('samples/realpuzzle.jpg')

		lowThreshold = 30
		ratio = 3
		kernel_size = 3
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = cv2.GaussianBlur(gray,(3,3),6)
		img = cv2.Canny(img,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
		cv2.imshow('test', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def import_pieces(self, path_to_pieces):
		pass

	def preprocess_pieces(self):
		"""Detect the edges of the puzzle piece and remove background"""
		pass

if __name__ == '__main__':
	pass