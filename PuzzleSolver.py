import numpy as np
import scipy.spatial.distance as distance
import math
import cv2
import sys

def get_edges_test():
        """Takes an image and gives an array same dimension as image that has 1 if
        edge, 0 if not for each pixel."""
        img = cv2.imread('samples/sample1/NWpiece.jpeg')

        lowThreshold = 30
        ratio = 3
        kernel_size = 3
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(gray,(3,3),6)
        img = cv2.Canny(img,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def match_colors_test():
	img1 = cv2.imread('samples/sample1/NWpiece.jpeg')
	height1, width1 = len(img1), len(img1[0])
	img2 = cv2.imread('samples/sample1/SWpiece.jpeg')
        height2, width2 = len(img2), len(img2[0])
	
	def bin_colors(pixels):
		bintransform=np.zeros((15, 15, 15))
		size=len(pixels)
		for i in range(size):
			pixel=pixels[i]
			blue=int(math.ceil(pixel[0]/float(17))) - 1 
			green=int(math.ceil(pixel[1]/float(17))) - 1
			red=int(math.ceil(pixel[2]/float(17))) -1
#			if pixel[0] > 254  or pixel[1] > 254 or pixel[2] > 254:
#				print(blue, green, red)
#				print(bintransform[blue, green, red])
			bintransform[blue, green, red] += 1
		return bintransform
	bin_colors(img1[height1-1])	
	edges=((bin_colors(img1[height1-1]), 1),(bin_colors(img2[height2-1]), 2),(bin_colors(img1[:, width1-1]), 1),(bin_colors(img2[:, width2-1]), 2),(bin_colors(img1[0]), 1),(bin_colors(img2[0]), 2), (bin_colors(img1[:, 0]), 1),(bin_colors(img2[:, 0]), 2))
#	print(edges[0][0])
	print(distance.cdist([np.reshape(edges[0][0], -1)], [np.reshape(edges[1][0], -1)]))
	edges1=[np.reshape(bin_colors(img1[height1-1]), -1), np.reshape(bin_colors(img1[0]), -1), np.reshape(bin_colors(img1[:, width1 - 1]), -1), np.reshape(bin_colors(img1[:, 0]), -1)]
	edges2=[np.reshape(bin_colors(img2[height2-1]), -1), np.reshape(bin_colors(img2[0]), -1), np.reshape(bin_colors(img2[:, width2 - 1]), -1), np.reshape(bin_colors(img2[:, 0]), -1)]
	print(distance.cdist(edges1, edges2))
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

	def import_pieces(self, path_to_pieces):
		pass

	def preprocess_pieces(self):
		"""Detect the edges of the puzzle piece and remove background"""
		pass

if __name__ == '__main__':
# 	get_edges_test()
	match_colors_test()
