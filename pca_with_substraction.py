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


# Flächenschwerpunkt berechnen
def calcCentroid(img):
    moments = cv2.moments(img, False)

    cX = float(moments["m10"] / moments["m00"])
    cY = float(moments["m01"] / moments["m00"])

    return [cX, cY]


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

	eigenvectors = pca.components_
	eigenvalues = pca.explained_variance_

	eigenvector_p1 = [eigenvectors[0,0], eigenvectors[0,1]]
	img_vector = [img, eigenvector_p1]

	return img_vector

def angle(v1, v2):
    ang1 = np.arctan2(*v1[::-1])
    ang2 = np.arctan2(*v2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def substraction(img1, img2, img1CentroidCoordinate, img2CentroidCoordinate):
	img1 = translate(img1, -img1CentroidCoordinate[0]+300, -img1CentroidCoordinate[1]+400)
	img2 = translate(img2, -img2CentroidCoordinate[0]+300, -img2CentroidCoordinate[1]+400)
	subImg = cv2.subtract(invert_image(img1), invert_image(img2))
	return invert_image(subImg)

#Scene generieren
scene = generateScene()
scene2 = generateScene()

kernel = np.ones((5,5),np.uint8)

scene_out = cv2.erode(scene,kernel,iterations = 1)
scene_out2 = cv2.erode(scene2,kernel,iterations = 1)

scene_out = fillContour(scene_out)

cv2.imshow("Scene1", scene_out)

scene_out2 = fillContour(scene_out2)

cv2.imshow("Scene2", scene_out2)

output1 = rotate(scene_out, randint(0, 360))
output1 = translate(output1, randint(-150, 150), randint(-150, 150))

##Ausführbereich
[scene1, vector_angle] = pca(scene_out)
[scene2, vector_angle2] = pca(scene_out2)
[output1, vector_angle_rotate_1] = pca(output1)

cv2.imshow('output1', output1)

#vertical = [0,1]
output1_rotate = rotate(output1, angle(vector_angle_rotate_1, vector_angle))
output1_rotate_add180 = rotate(output1, 180 + angle(vector_angle_rotate_1, vector_angle))

output1_rotate_2 = rotate(output1, angle(vector_angle_rotate_1, vector_angle2))
output1_rotate_add180_2 = rotate(output1, 180 + angle(vector_angle_rotate_1, vector_angle2))

sceneCentroidCoordinate = calcCentroid(invert_image(scene_out))
scene2CentroidCoordinate = calcCentroid(invert_image(scene_out2))

output1_rotate_CentroidCoordinate = calcCentroid(invert_image(output1_rotate))
output1_rotate_add180_CentroidCoordinate = calcCentroid(invert_image(output1_rotate_add180))

output1_rotate_2_CentroidCoordinate = calcCentroid(invert_image(output1_rotate_2))
output1_rotate_add180_2_CentroidCoordinate = calcCentroid(invert_image(output1_rotate_add180_2))

sub1 = substraction(output1_rotate, scene1, output1_rotate_CentroidCoordinate, sceneCentroidCoordinate)
sub1_add180 = substraction(output1_rotate_add180, scene1, output1_rotate_add180_CentroidCoordinate, sceneCentroidCoordinate)

sub2 = substraction(output1_rotate_2, scene2, output1_rotate_2_CentroidCoordinate, scene2CentroidCoordinate)
sub2_add180 = substraction(output1_rotate_add180_2, scene2, output1_rotate_add180_2_CentroidCoordinate, scene2CentroidCoordinate)

cv2.imshow('sub1', sub1)
cv2.imshow('sub1_add180', sub1_add180)
cv2.imshow('sub2', sub2)
cv2.imshow('sub2_add180', sub2_add180)

scene1_pixel = cv2.countNonZero(scene_out)
scene2_pixel = cv2.countNonZero(scene_out2)

count1 = cv2.countNonZero(sub1)
count1add180 = cv2.countNonZero(sub1_add180)

count2 = cv2.countNonZero(sub2)
count2add180 = cv2.countNonZero(sub2_add180)

print('Scene_1 schwarz Pixeln =',800*600-scene1_pixel)
print('Scene_2 schwarz Pixeln =',800*600-scene2_pixel)
print('Substraction_1 schwarz Pixeln =',800*600-count1)
print('Substraction_1_add180 schwarz Pixeln =',800*600-count1add180)
print('Substraction_2 schwarz Pixeln =',800*600-count2)
print('Substraction_2_add180 schwarz Pixeln =',800*600-count2add180)

cv2.waitKey(0)
cv2.destroyAllWindows()
