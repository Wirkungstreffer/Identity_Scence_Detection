#!/usr/bin/python3

# commit and push to git with: git add . ; git commit -m "Next commit3" ; git push origin master

from __future__ import print_function
from __future__ import division
from math import atan2, cos, sin, sqrt, pi


from os import listdir
import numpy as np


from random import randint
from mahotas import features
from sklearn.decomposition import PCA
import math
import cv2


print("x"*50)
print("Image Generator")
print("x"*50)
print()


#Funktionsdefinitionen

def invert_image(img):
	return (255-img)

def add_images(img1, img2):
	temp = invert_image(img1) + invert_image(img2)
	return invert_image(temp)

def add_images4(img1, img2, img3, img4):
	temp = invert_image(img1) + invert_image(img2) + invert_image(img3) + invert_image(img4)
	return invert_image(temp)

def translate(img, x, y):
	rows, cols = img.shape

	M = np.float32([[1,0,x], [0,1,y]])
	img = cv2.warpAffine(img, M, (cols,rows), borderMode = cv2.BORDER_REPLICATE)

	return img

def rotate(img, winkel):
	rows, cols = img.shape

	# Argumente: Center, Angle, Scale
	M = cv2.getRotationMatrix2D((cols/2,rows/2),winkel,1)
	img = cv2.warpAffine(img, M, (cols,rows), borderMode = cv2.BORDER_REPLICATE)

	return img


# Generate Scene
def generateScene():
    line_styles = ["dreieck", "ellipse", "gerade", "rechteck"]
    orientierung = ["links", "oben", "rechts", "unten"]
    path0 = ["", "", "", ""]
    path1 = ["", "", "", ""]

    for num in range(4):
        path0[num] = "Kanten/" + "kante_" + orientierung[num] + "_" + line_styles[randint(0, 3)] + ".png"

    for num in range(4):
        path1[num] = "Kanten/" + "kante_" + orientierung[num] + "_" + line_styles[randint(0, 3)] + ".png"

    path1[1] = path0[3]

    img0 = []
    img1 = []

    for num in range(4):
        img0.append(cv2.imread(path0[num], 0))
        img1.append(cv2.imread(path1[num], 0))

    img1[1] = translate(img1[1], 0, -100)

    obj0 = add_images4(img0[0], img0[1], img0[2], img0[3])
    obj1 = add_images4(img1[0], img1[1], img1[2], img1[3])

    obj1 = translate(obj1, 0, 100)

    return add_images(obj0,obj1)


def get_huMoments(img, name):
	# Hu-Momente
	moments = cv2.moments(img, False)
	huMoments = cv2.HuMoments(moments)

	# log Transformation
	for i in range(0,7):
		huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
	print()
	print("HuMoments (log corrected) from "+ name +": {}".format(huMoments))

	return huMoments


def get_matchShapes(img0, img1, name0, name1):
	# MatchShapes
	img0 = invert_image(img0)
	img1 = invert_image(img1)
	contours_match = [0, 0, 0]
	contours_match[0] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I1, 0)
	contours_match[1] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I2, 0)
	contours_match[2] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I3, 0)
	print()
	print("ContoursMatch of "+ name0 +" and "+ name1 +": {}".format(contours_match))

	return contours_match


# Flächenschwerpunkt berechnen
def calcCentroid(img):
    moments = cv2.moments(img, False)

    cX = float(moments["m10"] / moments["m00"])
    cY = float(moments["m01"] / moments["m00"])

    print()
    print("Flächenschwerpunkt: {}, {}".format(cX, cY))

    return [cX, cY]


# Zernike Moments
def get_zernikeMoments(img, name):
	ordnung = 8
	radius = 200
	zernike = features.zernike_moments(img, radius, ordnung)
	print()
	print("Zernike Momente von "+name+" der Ordnung {}, Radius in px {}: {}".format(ordnung, radius, zernike))

	return zernike

def getZernikeMatchShapes(img0, img1, name0, name1):
	img0 = invert_image(img0)
	img1 = invert_image(img1)
	ordnung = 8
	radius = 200
	zernike1 = features.zernike_moments(img0, radius, ordnung)
	zernike2 = features.zernike_moments(img1, radius, ordnung)

	zernike_contours_match1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for k in range(0, 24):
		zernike_contours_match1[k] = abs(1/(zernike1[k]) - 1/(zernike2[k]))

	zernike_contours_match2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for w in range(0, 24):
		zernike_contours_match2[w] = abs(zernike1[w] - zernike2[w])

	zernike_contours_match3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for f in range(0, 24):
		zernike_contours_match3[f] = (abs(zernike1[f] - zernike2[f]))/(abs(zernike1[f]))

	zernike_match = [0, 0, 0]
	zernike_match[0]= np.sum(zernike_contours_match1)
	zernike_match[1]= np.sum(zernike_contours_match2)
	zernike_match[2]= np.sum(zernike_contours_match3)

	print("\nZernikeContoursMatch of "+ name0 +" and "+ name1 +": {}".format(zernike_match))

# PCA Achsen Zeichnen
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

def fillContour(img):
	# Threshold.
	# Set values equal to or above 220 to 0.
	# Set values below 220 to 255.

	th, im_th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV);

	# Copy the thresholded image.
	im_floodfill = im_th.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_th.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	img_out = im_th | im_floodfill_inv

	return invert_image(img_out)

#Ausführbereich

scene = generateScene()
scene2 = generateScene()

kernel = np.ones((5,5),np.uint8)

scene_out = cv2.erode(scene,kernel,iterations = 1)
scene_out2 = cv2.erode(scene2,kernel,iterations = 1)

scene_out = fillContour(scene_out)

cv2.imshow("Scene1", scene_out)

scene_out2 = fillContour(scene_out2)

cv2.imshow("Scene2", scene_out2)
#cv2.imshow("Scene2", scene2)

get_huMoments(scene_out, "scene_out")
sceneCentroidCoordinate = calcCentroid(invert_image(scene_out))
get_zernikeMoments(scene_out, "Scene")

output1 = rotate(scene_out, randint(0, 360))
output1 = translate(output1, randint(-150, 150), randint(-150, 150))

get_huMoments(output1, "output1")
output1CentroidCoordinate = calcCentroid(invert_image(output1))
get_zernikeMoments(output1, "Output1")

output2 = rotate(scene_out, randint(0, 360))
output2 = translate(output2, randint(-150, 150), randint(-150, 150))

get_huMoments(output2, "output2")
output2CentroidCoordinate = calcCentroid(invert_image(output2))
get_zernikeMoments(output2, "Output2")

#get_zernikeMoments(scene2, "Scene2")

get_matchShapes(scene_out, output1, "scene1", "output1")
get_matchShapes(scene_out, scene_out2, "scene1", "scene2")

def pca(img):
    
	# Convert image to binary
	_, bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	# PCA for everything
	contour_size = 0

	for i, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		# Ignore contours that are too small or too large
		if area < 1e2 or 1e5 < area:
			continue

		contour_size += len(contour)

	all_data_pts = np.empty((contour_size, 2), dtype=np.float64)


	k = 0

	for i, contour in enumerate(contours):

		area = cv2.contourArea(contour)
		# Ignore contours that are too small or too large
		if area < 1e2 or 1e5 < area:
		    continue

		sz = len(contour)
		data_pts = np.empty((sz, 2), dtype=np.float64)

		# cv2.drawContours(scene, contours, i, (0, 255, 0), 3)

		for i in range(data_pts.shape[0]):
			data_pts[i,0] = contour[i,0,0]
			data_pts[i,1] = contour[i,0,1]

			all_data_pts[i+k,0] = contour[i,0,0]
			all_data_pts[i+k,1] = contour[i,0,1]

		k += len(contour)

	# import sys
	# np.set_printoptions(threshold=sys.maxsize)
	# print("{}".format(all_data_pts))

	# Perform PCA analysis
	mean = np.empty((0))
	mean, eigenvectors, eigenvalues = cv2.PCACompute2(all_data_pts, mean)

	rows, cols = img.shape
	count = 0
	for y in range(rows):
		for x in range(cols):
			if bw[y, x] <= 50:
				count += 1

	a = np.empty((count, 2), dtype=np.float64)

	cc = 0
	for y in range(rows):
		for x in range(cols):
			if bw[y, x] <= 50:
				a[cc, 0] = x
				a[cc, 1] = y
				cc += 1

	# PCA mit sklearn
	pca = PCA(n_components=2)
	pca.fit(a)

	print(pca.components_)
	print(pca.explained_variance_)

	eigenvectors = pca.components_
	eigenvalues = pca.explained_variance_

	eigenvector_p1 = [eigenvectors[0,0], eigenvectors[0,1]]
	img_vector = [img, eigenvector_p1]

	return img_vector

def angle(v1, v2):
    ang1 = np.arctan2(*v1[::-1])
    ang2 = np.arctan2(*v2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def centralizeOutputs(img1, img2, img1CentroidCoordinate, img2CentroidCoordinate):
    img1 = translate(img1, -img1CentroidCoordinate[0]+300, -img1CentroidCoordinate[1]+400)
    #print(-img1CentroidCoordinate[0]+300)
    #print(-img1CentroidCoordinate[1]+400)
    img2 = translate(img2, -img2CentroidCoordinate[0]+300, -img2CentroidCoordinate[1]+400)
    #print(-img2CentroidCoordinate[0]+300)
    #print(-img2CentroidCoordinate[1]+400)
    resultImg = add_images(img1, img2)

    cv2.imshow('result', resultImg)

[scene1, vector_angle] = pca(scene_out)
[output1, vector_angle1] = pca(output1)
[output2, vector_angle2] = pca(output2)

cv2.imshow('output1', output1)
cv2.imshow('output2', output2)

output1_rotate = rotate(output1, angle(vector_angle1, vector_angle))
output2_rotate = rotate(output2, angle(vector_angle2, vector_angle))

cv2.imshow('output1_rotate', output1_rotate)
cv2.imshow('output2_rotate', output2_rotate)

output1_rotate_CentroidCoordinate = calcCentroid(invert_image(output1_rotate))
output2_rotate_CentroidCoordinate = calcCentroid(invert_image(output2_rotate))

get_matchShapes(scene1, output1_rotate, "scene1", "output1_rotate")

centralizeOutputs(output1_rotate, scene1, output1_rotate_CentroidCoordinate, sceneCentroidCoordinate)

output1_center = translate(output1, -output1CentroidCoordinate[0]+300, -output1CentroidCoordinate[1]+400)
getZernikeMatchShapes(scene_out, output1_center, "Scene1", "Output1")
getZernikeMatchShapes(scene_out, scene_out2, "Scene1", "Scene2")


cv2.waitKey(0)
cv2.destroyAllWindows()
