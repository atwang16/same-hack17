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
	img2 = cv2.imread('samples/sample1/SEpiece.jpeg')
        height2, width2 = len(img2), len(img2[0])

	images = (cv2.imread('samples/sample1/NWpiece.jpeg'), cv2.imread('samples/sample1/NEpiece.jpeg'), cv2.imread('samples/sample1/SWpiece.jpeg'), cv2.imread('samples/sample1/SEpiece.jpeg'))
	
	edgesfinal=[np.reshape(bin_colors(img1[height1-1]), -1), np.reshape(bin_colors(img1[0]), -1), np.reshape(bin_colors(img1[:, width1 - 1]), -1), np.reshape(bin_colors(img1[:, 0]), -1), np.reshape(bin_colors(img2[height2-1]), -1), np.reshape(bin_colors(img2[0]), -1), np.reshape(bin_colors(img2[:, width2 - 1]), -1), np.reshape(bin_colors(img2[:, 0]), -1)]
	sixteenedges = [[0 for t in range(4*len(images))] for s in range(4*len(images))]
	for i in range(len(images)):
		sixteenedges[4*i] = np.reshape(bin_colors(images[i][0]), -1)
		sixteenedges[4*i+1] = np.reshape(bin_colors(images[i][:, len(images[i][0]) - 1]), -1)
		sixteenedges[4*i+2] = np.reshape(bin_colors(images[i][len(images[i]) - 1]), -1)
		sixteenedges[4*i+3] = np.reshape(bin_colors(images[i][:, 0]), -1)
	print(sixteenedges)



	distances= distance.cdist(sixteenedges, sixteenedges)
	# Set impossible edge matchups to infinite
	for i in range(len(distances)):
		number=i%4
		distances[i][i]= np.inf
		for j in range(number):
			distances[i][i-j-1] = np.inf
			distances[i-j-1][i] = np.inf
				
#	print(distances)
	steps = [0 for i in range(len(distances))]
	stepnum=0
	while (np.min(distances) != np.inf):
		indices = np.unravel_index(np.argmin(distances), distances.shape)
		print(indices)
		steps[stepnum] = indices
		stepnum += 1
		distances[indices[0]] = np.inf
		distances[:, indices[0]] = np.inf
		distances[indices[1]] = np.inf
		distances[:, indices[1]] = np.inf

class PuzzleSolver():
	def __init__(self):
		self.pieces = list()
	
	def import_pieces(self, path_to_pieces):
		pass

	def get_front_binary_image(self, img):
	"""
	TODO: write description

	:param img: an image with a white background
	:return:
	"""
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert from color to grayscale
	img_gray_gauss = cv2.GaussianBlur(img_gray, (5, 5), 0) # apply Gaussian blur
	th, img_gray_thresh = cv2.threshold(img_gray_gauss, 245, 255, cv2.THRESH_BINARY) # threshold image so relevant part becomes black
 	return cv2.bitwise_not(img_gray_thresh)

	def get_back_binary_image(self, img):
	    """
	    TODO: write description

	    :param img: an image with a white background
	    :return:
	    """
	    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert from color to grayscale
	    img_gray_gauss = cv2.GaussianBlur(img_gray, (5, 5), 0) # apply Gaussian blur
	    th, img_gray_thresh = cv2.threshold(img_gray_gauss, 100, 255, cv2.THRESH_BINARY) # threshold image so relevant part becomes black
	    return cv2.flip(img_gray_thresh, 1)


	def get_edges(self, bin_img):
	    return cv2.Canny(bin_img, 60, 100)


	def get_com(self, img):
	    """
	    Get center of mass (i.e. intensity) of image.

	    :param img:
	    :return:
	    """
	    M = cv2.moments(img)
	    cx = int(M['m10'] / M['m00'])
	    cy = int(M['m01'] / M['m00'])
	    return cx, cy

	def get_convexity(self, img, visualize=False):
		np.set_printoptions(threshold=np.nan)
		"""Returns convexity of each edge from top edge going clockwise.

		1 = convex
		0 = concave
		-1 = edge
		"""
		edges = get_edges(get_back_binary_image(img))
		# edges = get_edges(img)

		if visualize:
			cv2.imshow('image', edges)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		tl, tr, br, bl = get_corners(img)

		top_edge = int((tl[1] + tr[1]) / 2)
		bottom_edge = int((br[1] + bl[1]) / 2)
		left_edge = int((tl[0] + bl[0]) / 2)
		right_edge = int((tr[0] + br[0]) / 2)

		convexity = list()
		margin = 10
		# Top edge
		if sum(edges[top_edge - margin]) > 1 * 255:
			convexity.append(1)
		elif sum(edges[top_edge + margin]) > 3 * 255:
			convexity.append(0)
		else:
			convexity.append(-1)

		# Right edge
		if sum(edges[:, right_edge - margin]) > 3 * 255:
			convexity.append(0)
		elif sum(edges[:, right_edge + margin]) > 1 * 255:
			convexity.append(1)
		else:
			convexity.append(-1)

		# Bottom edge
		if sum(edges[bottom_edge - margin]) > 3 * 255:
			convexity.append(0)
		elif sum(edges[bottom_edge + margin]) > 1 * 255:
			convexity.append(1)
		else:
			convexity.append(-1)

		# Left edge
		if sum(edges[:, left_edge - margin]) > 1 * 255:
			convexity.append(1)
		elif sum(edges[:, left_edge + margin]) > 3 * 255:
			convexity.append(0)
		else:
			convexity.append(-1)

		return convexity

	def get_corners(self, img, visualize=False):
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
