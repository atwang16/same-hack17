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
        return translated

    def apply_mask(self, img_front, img_back):
        min_score = np.inf
        best_img_back = None

        for a in range(0, 360, 1):
            cx_f, cy_f = self.get_com(img_front)
            cx_b, cy_b = self.get_com(img_back)
            img_back_rot = self.rotate_image(img_back, (cx_f, cy_f), a, img_front.shape)

            overlay = img_front ^ img_back_rot
            xor_sum = np.sum(overlay)
            if xor_sum < min_score:
                min_score = xor_sum
                best_img_back = img_back_rot

        return best_img_back
    
    def get_corners(self, img, visualize=False):
            """
            Detects the corners of the puzzle piece.

            RETURN
            A list of python lists of the x and y coordinates beginning with the top-left and going clockwise
            """
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(img)
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

            return [tl, tr, br, bl]
    
    def get_convexity(self, binary_img, visualize=False):
        """Returns convexity of each edge from top edge going clockwise.

        1 = convex
        0 = concave
        -1 = edge
        """
        edges = self.get_edges(binary_img)

        if visualize:
            cv2.imshow('image', edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        tl, tr, br, bl = self.get_corners(binary_img)

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
        for index, piece in enumerate(self.back_binary_images):
            convexities = self.get_convexity(piece)
            for edge_index, edge_status in enumerate(convexities):
                if edge_status == 0:
                    self.concave_edges.append(index * 4 + edge_index)
                elif edge_status == 1:
                    self.convex_edges.append(index * 4 + edge_index)
                else:
                    self.straight_edges.append(index * 4 + edge_index)

    def get_all_corners(self):
        for image in self.back_binary_images:
            self.corners.extend(self.get_corners(image))

    def get_all_piece_dimensions(self):
        for i in range(self.num_pieces):
            corners = self.corners[i * 4:i * 4 + 4]
            top = corners[1][0] - corners[0][0]
            right = corners[2][1] - corners[1][1]
            bottom = corners[2][0] - corners[3][0]
            left = corners[3][1] - corners[0][1]
        self.piece_dim.extend([top, right, bottom, left])

    def get_color_histogram(self, side):
        nbins = 15
        bintransform = np.zeros((nbins, nbins, nbins))
        for i in range(len(side)):
            pixel = side[i]
            blue = int(math.ceil(pixel[0] / float(255 / nbins))) - 1
            green = int(math.ceil(pixel[1] / float(255 / nbins))) - 1
            red = int(math.ceil(pixel[2] / float(255 / nbins))) - 1
            bintransform[blue, green, red] += 1
        return bintransform

    def get_color_descriptors(self, img_front_co, img_front_bw, img_back_bw, corners):
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
        mask, angle, trans = self.apply_mask(img_front_bw, img_back_bw)
        mask_edge = self.get_edges(mask)
        mask_edge = mask_edge[:, :, np.newaxis] # add a third dimension for broadcasting
        img_front_boundary = mask_edge & img_front_co

        angle *= np.pi / 180 # convert to radians
        corners = np.concatenate((corners, np.ones((4, 1))), axis=1)
        H = np.array([np.cos(angle), -np.sin(angle), trans[0]],
                     [np.sin(angle),  np.cos(angle), trans[1]])
        tf_corners = corners * H.T
        tl = (tf_corners[0, 0], tf_corners[0, 1])
        tr = (tf_corners[1, 0], tf_corners[1, 1])
        br = (tf_corners[2, 0], tf_corners[2, 1])
        bl = (tf_corners[3, 0], tf_corners[3, 1])

        def tlbr_diagonal(x, y):
            m = (tl[1] - br[1]) / (tl[0] - br[0])
            return m * (x - tl[0]) + (tl[1] - y)

        def trbl_diagonal(x, y):
            m = (tr[1] - bl[1]) / (tr[0] - bl[0])
            return m * (x - tr[0]) + (tr[1] - y)

        sides = [[] for _ in range(4)]

        for x in xrange(img_front_boundary.shape[0]):
            for y in xrange(img_front_boundary.shape[1]):
                c = list(img_front_boundary[x, y, :])
                if c != [0, 0, 0]:
                    t1 = tlbr_diagonal(x, y)
                    t2 = trbl_diagonal(x, y)
                    if t1 >= 0 and t2 >= 0:
                        sides[0].append(c)
                    elif t1 >= 0 and t2 < 0:
                        sides[1].append(c)
                    elif t1 < 0 and t2 < 0:
                        sides[2].append(c)
                    else:
                        sides[3].append(c)

        return [np.reshape(self.get_color_histogram(sides[0]), -1),
                np.reshape(self.get_color_histogram(sides[1]), -1),
                np.reshape(self.get_color_histogram(sides[2]), -1),
                np.reshape(self.get_color_histogram(sides[3]), -1)]

    def get_all_descriptors(self):
        for i in range(self.num_pieces)
        self.color_descriptors

if __name__ == '__main__':
    pass
