import numpy as np
import scipy.spatial.distance as distance
import math
import cv2
import sys

def match_colors_test():
    img1 = cv2.imread('samples/sample1/NWpiece.jpeg')
    height1, width1 = len(img1), len(img1[0])
    img2 = cv2.imread('samples/sample1/SEpiece.jpeg')
    height2, width2 = len(img2), len(img2[0])

    images = (cv2.imread('samples/sample1/NWpiece.jpeg'), cv2.imread('samples/sample1/NEpiece.jpeg'), cv2.imread('samples/sample1/SWpiece.jpeg'), cv2.imread('samples/sample1/SEpiece.jpeg'))

    def bin_colors(pixels):
        bintransform=np.zeros((15, 15, 15))
        size=len(pixels)
        for i in range(size):
            pixel=pixels[i]
            blue=int(math.ceil(pixel[0]/float(17))) - 1 
            green=int(math.ceil(pixel[1]/float(17))) - 1
            red=int(math.ceil(pixel[2]/float(17))) -1
#           if pixel[0] > 254  or pixel[1] > 254 or pixel[2] > 254:
#               print(blue, green, red)
#               print(bintransform[blue, green, red])
            bintransform[blue, green, red] += 1
        return bintransform

    edgesfinal=[np.reshape(bin_colors(img1[height1-1]), -1), np.reshape(bin_colors(img1[0]), -1), np.reshape(bin_colors(img1[:, width1 - 1]), -1), np.reshape(bin_colors(img1[:, 0]), -1), np.reshape(bin_colors(img2[height2-1]), -1), np.reshape(bin_colors(img2[0]), -1), np.reshape(bin_colors(img2[:, width2 - 1]), -1), np.reshape(bin_colors(img2[:, 0]), -1)]
    sixteenedges = [[0 for t in range(4*len(images))] for s in range(4*len(images))]
    for i in range(len(images)):
        sixteenedges[4*i] = np.reshape(bin_colors(images[i][0]), -1)
        sixteenedges[4*i+1] = np.reshape(bin_colors(images[i][:, len(images[i][0]) - 1]), -1)
        sixteenedges[4*i+2] = np.reshape(bin_colors(images[i][len(images[i]) - 1]), -1)
        sixteenedges[4*i+3] = np.reshape(bin_colors(images[i][:, 0]), -1)
    print(sixteenedges)

    result=generate_steps(sixteenedges, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
    print(result)
#     distances= distance.cdist(sixteenedges, sixteenedges)
    # Set impossible edge matchups to infinite
#     for i in range(len(distances)):
#         number=i%4
#         distances[i][i]= np.inf
#         for j in range(number):
#             distances[i][i-j-1] = np.inf
#             distances[i-j-1][i] = np.inf
# 
# #	print(distances)
#     steps = [0 for i in range(len(distances))]
#     stepnum=0
#     while (np.min(distances) != np.inf):
#         indices = np.unravel_index(np.argmin(distances), distances.shape)
#         print(indices)
#         steps[stepnum] = indices
#         stepnum += 1
#         distances[indices[0]] = np.inf
#         distances[:, indices[0]] = np.inf
#         distances[indices[1]] = np.inf
#         distances[:, indices[1]] = np.inf


def generate_steps(edges,convex, concave):
# Assuming edges is a list of lists of pixels (one list of pixels in BGR for each edge)
    points=[]
    def bin_colors(pixels):
        bintransform=np.zeros((15, 15, 15))
        size=len(pixels)
        for i in range(size):
            pixel=pixels[i]
            blue=int(math.ceil(pixel[0]/float(17))) - 1
            green=int(math.ceil(pixel[1]/float(17))) - 1
            red=int(math.ceil(pixel[2]/float(17))) -1
#           if pixel[0] > 254  or pixel[1] > 254 or pixel[2] > 254:
#               print(blue, green, red)
#               print(bintransform[blue, green, red])
            bintransform[blue, green, red] += 1
        return bintransform

    for i in convex:
        for j in concave:
            if (not(j <= i<= j+ (j%4)) and not (i <= j <= i+(i%4))):
                points.append(((i, j), distance.cdist([np.reshape(bin_colors(edges[i]))],[np.reshape(bin_colors(edges[j]))])[0][0]))
#   print(points)
    approx=sorted(points, key=lambda x: x[1])
    used=set()
    steps=[]
    limit = 100 # First 100 steps of puzzle will be given
    stepcount=0

    for l in range(len(approx)):
        coordinates=approx[l][0]
        if coordinates[0] not in used and coordinates[1] not in used:
            stepcount +=1
            steps.append[coordinates]
            set.add(coordinates[0])
            set.add(coordinates[1])
        if stepcount > limit:
            break
    return steps 

if __name__ == '__main__':
    match_colors_test()
