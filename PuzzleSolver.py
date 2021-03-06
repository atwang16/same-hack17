import numpy as np
import math, cv2, sys, glob

class PuzzleSolver():
    def __init__(self):
        # Color images
        self.front_images = list()
        self.back_images = list()
        # B/W binary images
        self.front_binary_images = list()
        self.back_binary_images = list()

        self.corners = list()
        self.piece_dim = list()
        self.convex_edges = list()
        self.concave_edges = list()
        self.straight_edges = list()
        self.color_descriptors = list()

    def import_pieces(self, path_to_pieces):
        """
        Runs methods to import all images and preprocess them, including determining the convexity of each side,
        finding corners, calculating piece dimensions, and computing color descriptors.

        :param path_to_pieces: path to directory containing images of puzzle pieces. Pieces must end in '_front.jpg'
        or '_back.jpg,' with the first part of the filename identical for the same piece.
        :return: None
        """
        front_files = sorted(glob.glob(path_to_pieces + '/*_front.jpg'))
        back_files = sorted(glob.glob(path_to_pieces + '/*_back.jpg'))

        self.num_pieces = len(back_files)
        # Load color images
        self.front_images = [cv2.imread(piece) for piece in front_files]
        self.back_images = [cv2.imread(piece) for piece in back_files]
        # Process images into binary images
        self.front_binary_images = [self.get_front_binary_image(piece) for piece in self.front_images]
        self.back_binary_images = [self.get_back_binary_image(piece) for piece in self.back_images]
        # Get basic metrics
        self.sort_convexities()
        self.get_all_corners()
        self.get_all_piece_dimensions()
        self.get_all_descriptors()

        return None

    def get_front_binary_image(self, img):
        """
        Processes a front image of a puzzle piece to form a binary image (white puzzle piece, black background) through
        a Gaussian blur and binary threshold.

        :param img: an image of the front of a puzzle piece with a white background, in numpy array format
        :return: a binary image with a white puzzle piece and black background
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert from color to grayscale
        img_gray_gauss = cv2.GaussianBlur(img_gray, (5, 5), 0) # apply Gaussian blur
        th, img_gray_thresh = cv2.threshold(img_gray_gauss, 245, 255, cv2.THRESH_BINARY) # threshold image so relevant part becomes black
        return cv2.bitwise_not(img_gray_thresh)

    def get_back_binary_image(self, img):
        """
        Processes a back image of a puzzle piece to form a binary image (white puzzle piece, black background) through
        a Gaussian blur and binary threshold.

        :param img: an image of the back of a puzzle piece with a black background, in numpy array format
        :return: a binary image with a white puzzle piece and black background
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert from color to grayscale
        img_gray_gauss = cv2.GaussianBlur(img_gray, (21, 21), 0) # apply Gaussian blur
        th, img_gray_thresh = cv2.threshold(img_gray_gauss, 200, 255, cv2.THRESH_BINARY) # threshold image so relevant part becomes black
        img_gray_thresh = cv2.bitwise_not(img_gray_thresh)
        return cv2.flip(img_gray_thresh, 1)

    def get_edges(self, bin_img):
        """
        Runs an edge detection routine on an image based on the Canny edge detector.

        :param bin_img: a binary image in numpy array format
        :return: the edges of a binary image, in numpy array format
        """
        return cv2.Canny(bin_img, 60, 100)

    def get_com(self, bin_img):
        """
        Get center of mass (i.e. intensity) of image using moments.

        :param bin_img: a binary image in numpy array format
        :return: the coordinates of the center of mass, as a tuple
        """
        M = cv2.moments(bin_img)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy

    def rotate_image(self, img, desired_center, angle_deg, final_dim):
        """
        Rotate an image by a specified angle, translate to desired center, and crop to desired dimension.

        :param img: image file, in the form of a numpy array
        :param desired_center: the desired coordinates for the COM of the image
        :param angle_deg: the angle (in degrees) over which to rotate the image
        :param final_dim: the final dimensions of the returned image
        :return: an image with the specified transformations
        """
        cx, cy = self.get_com(img)
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M,
                                 (final_dim[1], final_dim[0]))  # not sure why it has to be reversed, but it does
        cx_new, cy_new = self.get_com(rotated)
        delta = (desired_center[0] - cx_new, desired_center[1] - cy_new)
        translated = cv2.warpAffine(rotated, np.float32([[1, 0, delta[0]], [0, 1, delta[1]]]),
                                    (final_dim[1], final_dim[0]))
        return translated, (desired_center[0] - cx, desired_center[1] - cy)

    def apply_mask(self, front_bin_img, back_bin_img):
        """
        Applies mask of back image to front image of puzzle piece in order to provide a more distinct edge to the piece.
        The mask is applied by aligning the center of masses of the front and back and then minimizing the sum of the XORs
        of the two images overlaid on each other (i.e. minimizing the amount of non-overlap), over a discretized range of
        possible angles. Accuracy in selecting the angle is guaranteed within a 1 degree margin.

        :param front_bin_img: a binary image of the front of a puzzle piece, in numpy array format
        :param back_bin_img: a binary image of the back of a puzzle piece, in numpy array format
        :return: the best transformed version of the back image, the associated angle, and the associated translation
        """
        min_score = np.inf
        best_img_back = None
        best_angle = None
        best_delta = None

        for a in range(0, 360, 1):
            cx_f, cy_f = self.get_com(front_bin_img)
            img_back_rot, delta = self.rotate_image(back_bin_img, (cx_f, cy_f), a, front_bin_img.shape)

            overlay = front_bin_img ^ img_back_rot
            xor_sum = np.sum(overlay)
            if xor_sum < min_score:
                min_score = xor_sum
                best_img_back = img_back_rot
                best_angle = a
                best_delta = delta

        return best_img_back, best_angle, best_delta
    
    def get_corners(self, img, visualize=True):
            """
            Detects the corners of the puzzle piece using a corner detection algorithm.

            :param img: a (binary) grayscale image in numpy array format
            :param visualize: an optional parameter to visualize the locations of the corners
            :return: A list of python lists of the x and y coordinates beginning with the top-left and going clockwise
            """

            gray = np.float32(img)
            gray = cv2.GaussianBlur(gray,(3,3),0)
            # _, gray = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 100, useHarrisDetector=True)
            corners = np.int0(corners)

            # Clean up output
            corner_list = list()
            for i in corners:
                x, y = i.ravel()
                corner_list.append([x, y])
                if visualize:
                    cv2.circle(img, (x, y), 3, (128, 0, 0), -1)
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

            return [tl, tr, br, bl]
    
    def get_convexity(self, bin_img, visualize=False):
        """
        Returns convexity of each edge (concave, convex, or straight) from top edge going clockwise, which will be one
        of the heuristics used to determine whether the sides of two puzzle pieces can match up.

        :param bin_img: a binary image, in the form of a numpy array
        :param visualize: an optional parameter to vizualize the results
        :return: the convexities of each side of a puzzle piece
        """
        """

        1 = convex
        0 = concave
        -1 = edge
        """
        edges = self.get_edges(bin_img)

        if visualize:
            cv2.imshow('image', edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        tl, tr, br, bl = self.get_corners(bin_img)

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

    def sort_convexities(self):
        """
        A wrapper function to sort the sides of the puzzle pieces by convexity (convex, concave, or straight).
        Each Python list stores the indices of the corresponding sides.

        :return: None
        """
        for index, piece in enumerate(self.back_binary_images):
            convexities = self.get_convexity(piece)
            for edge_index, edge_status in enumerate(convexities):
                if edge_status == 0:
                    self.concave_edges.append(index * 4 + edge_index)
                elif edge_status == 1:
                    self.convex_edges.append(index * 4 + edge_index)
                else:
                    self.straight_edges.append(index * 4 + edge_index)
        return None

    def get_all_corners(self):
        """
        A wrapper function to compute corners for every puzzle piece.

        :return: None
        """
        for image in self.back_binary_images:
            self.corners.extend(self.get_corners(image))
        return None

    def get_all_piece_dimensions(self):
        """
        Approximates and stores the length of each side using the corners identified for each puzzle piece.

        :return: None
        """
        for i in range(self.num_pieces):
            corners = self.corners[i * 4:i * 4 + 4]
            top = corners[1][0] - corners[0][0]
            right = corners[2][1] - corners[1][1]
            bottom = corners[2][0] - corners[3][0]
            left = corners[3][1] - corners[0][1]
            self.piece_dim.extend([top, right, bottom, left])
        return None

    def get_color_histogram(self, side):
        """
        Given a side of a puzzle piece, represented by a list of pixel colors, a descriptor is returned in the form of
        a 3D color histogram to allow for comparison of color patterns between the edges of different puzzle pieces.

        :param side: a Python list of lists of RGB colors (in the order BGR), representing the colors of all of the edge
        pixels along a particular side of a puzzle piece.
        :return: a 3D numpy array storing the histogram counts of the pixels along the side.
        """
        nbins = 15
        bintransform = np.zeros((nbins, nbins, nbins))
        for i in range(len(side)):
            pixel = side[i]
            blue = int(math.ceil(pixel[0] / float(255 / nbins))) - 1
            green = int(math.ceil(pixel[1] / float(255 / nbins))) - 1
            red = int(math.ceil(pixel[2] / float(255 / nbins))) - 1
            bintransform[blue, green, red] += 1
            # TODO: implement normalization here
        return bintransform

    def get_color_descriptors(self, img_front_co, img_front_bw, img_back_bw, visualize=False):
        """
        Returns a list of four color descriptors representing the color patterns of each of
        the four edges, extracted by applying a mask of the back of the puzzle piece to the
        front and using the pixels on the boundary to create a histogram description.

        :param img_front_co: The front of the image in color, represented by a numpy array.
        :param img_front_bw: The front of the image in black and white, represented by a numpy array.
        :param img_back_bw: The back of the image in black and white, represented by a numpy array.
        :param corners: The four corners of the base shape of the puzzle piece.
        :return: a list of four numpy arrays which are mathematical descriptors of the color patterns of each side of the puzzle piece.
        """
        thresh = 10

        mask, angle, trans = self.apply_mask(img_front_bw, img_back_bw)
        mask_edge = self.get_edges(mask)
        mask_edge_3ch = mask_edge[:, :, np.newaxis] # add a third dimension for broadcasting
        img_front_boundary = mask_edge_3ch & img_front_co

        corners = self.get_corners(mask)
        tl = tuple(corners[0])
        tr = tuple(corners[1])
        br = tuple(corners[2])
        bl = tuple(corners[3])

        if visualize:
            cv2.circle(img_front_boundary, tl, 3, 255, -1)
            cv2.circle(img_front_boundary, tr, 3, 255, -1)
            cv2.circle(img_front_boundary, br, 3, 255, -1)
            cv2.circle(img_front_boundary, bl, 3, 255, -1)

            cv2.imshow("front boundary", img_front_boundary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        np.set_printoptions(threshold=np.nan)

        def tlbr_diagonal(r, c):
            m = (tl[1] - br[1]) / (tl[0] - br[0])
            return (c - tl[1]) - m * (r - tl[0])

        def trbl_diagonal(r, c):
            m = (tr[1] - bl[1]) / (tr[0] - bl[0])
            return (c - tr[1]) - m * (r - tr[0])

        sides = [[] for _ in range(4)]

        mask_edge, contours, _ = cv2.findContours(mask_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        dist = [np.inf for _ in range(4)]
        corner_ind = [-1 for _ in range(4)]
        for i in range(len(contours[0])):
            pt = contours[0][i]
            for j in range(len(dist)):
                if dist[j] > (corners[j][0] - pt[0, 0])**2 + (corners[j][1] - pt[0, 1])**2:
                    dist[j] = (corners[j][0] - pt[0, 0])**2 + (corners[j][1] - pt[0, 1])**2
                    corner_ind[j] = i

        for i in range(len(contours[0])):
            pt = contours[0][i]
            if i <= corner_ind[0] or i > corner_ind[1]: # top side
                sides[0].append(img_front_boundary[pt[0, 1], pt[0, 0], :])
            elif corner_ind[0] < i <= corner_ind[3]: # right side
                sides[1].append(img_front_boundary[pt[0, 1], pt[0, 0], :])
            elif corner_ind[3] < i <= corner_ind[2]: # right side
                sides[2].append(img_front_boundary[pt[0, 1], pt[0, 0], :])
            elif corner_ind[2] < i <= corner_ind[1]: # right side
                sides[3].append(img_front_boundary[pt[0, 1], pt[0, 0], :])

        return [np.reshape(self.get_color_histogram(sides[0]), -1),
                np.reshape(self.get_color_histogram(sides[1]), -1),
                np.reshape(self.get_color_histogram(sides[2]), -1),
                np.reshape(self.get_color_histogram(sides[3]), -1)]

    def get_all_descriptors(self):
        for i in range(self.num_pieces):
            self.color_descriptors.extend(self.get_color_descriptors(self.front_images[i], self.front_binary_images[i],
                                                                     self.back_binary_images[i]))

    def is_same_piece(self, ind_1, ind_2):
        """Test if two edges are on the same piece"""
        # int division
        return ind_1 / 4 == ind_2 / 4

    def edge_len_match(self, ind_1, ind_2):
        """Test if two edge lengths are close enough"""
        return abs(self.piece_dim[ind_1] - self.piece_dim[ind_2]) < 30

    def solve(self):
        """The magic function that solves the puzzle"""
        edges = list()
        for convex_edge in self.convex_edges:
            for concave_edge in self.concave_edges:
                if not self.is_same_piece(convex_edge, concave_edge) and self.edge_len_match(convex_edge, concave_edge):
                    # print(self.color_descriptors[convex_edge])
                    # print(self.color_descriptors[concave_edge])
                    color_dist = np.linalg.norm(self.color_descriptors[convex_edge] - self.color_descriptors[concave_edge])
                    # color_dist = distance.cdist([np.reshape(bin_colors(edges[i]))],[np.reshape(bin_colors(edges[j]))])[0][0] # MIGHT NEED TO REPLACE WITH NUMPY
                    edges.append((convex_edge, concave_edge, color_dist))
        
        # Sort valid edges by smallest edge distance
        edges = sorted(edges, key=lambda x: x[2])

        # Generate steps (the n most-likely non-contradictory combinations)
        used_edges = set()
        steps = list()
        num_steps = self.num_pieces
        for step in range(num_steps):
            edge_1, edge_2, _ = edges[step]
            if edge_1 not in used_edges and edge_2 not in used_edges:
                steps.append((edge_1, edge_2))
                used_edges.add(edge_1)
                used_edges.add(edge_2)
        return steps

if __name__ == '__main__':
    pass
