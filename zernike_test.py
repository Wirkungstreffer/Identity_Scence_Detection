from random import randint
from mahotas import features
import cv2
import math
import numpy as np

print("x"*50 + "\nZernike-Moment-Test\n" + "x"*50 + "\n")

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

    #print()
    #print("Flächenschwerpunkt: {}, {}".format(cX, cY))

    return [cX, cY]


# Zernike Moments
def get_zernikeMoments(img, name):
	ordnung = 8
	radius = 200
	zernike = features.zernike_moments(img, radius, ordnung)
	print()
	print("Zernike Momente von "+name+" der Ordnung {}, Radius in px {}: {}".format(ordnung, radius, zernike))

	return zernike

def getZernikeMatchShapes(img0, img1, name0, name1, id):
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

	print("\n"+ id + ".ZernikeContoursMatch of "+ name0 +" and "+ name1 +": {}".format(zernike_match))

	return zernike_match


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

testNumber = int(input("Wie viele Tests sollen durchgeführt werden? "))

i = 0
check = 0

while(i<testNumber):

    scene = generateScene()
    scene2 = generateScene()

    kernel = np.ones((5,5),np.uint8)

    scene_out = cv2.erode(scene,kernel,iterations = 1)
    scene_out2 = cv2.erode(scene2,kernel,iterations = 1)

    scene_out = fillContour(scene_out)

    #cv2.imshow("Scene", scene_out)

    scene_out2 = fillContour(scene_out2)

    #cv2.imshow("Scene2", scene_out2)

    #get_zernikeMoments(scene_out, "Scene")

    output1 = rotate(scene_out, randint(0, 360))
    output1 = translate(output1, randint(-150, 150), randint(-150, 150))

    output1CentroidCoordinate = calcCentroid(invert_image(output1))
    #get_zernikeMoments(output1, "Output1")


    output1_center = translate(output1, -output1CentroidCoordinate[0]+300, -output1CentroidCoordinate[1]+400)
    contoursMatch1 = getZernikeMatchShapes(scene_out, output1_center, "Scene1", "Output1", str(i+1))
    contoursMatch2 = getZernikeMatchShapes(scene_out, scene_out2, "Scene1", "Scene2", str(i+1))

    check_old = check

    #if contoursMatch1[0]<=contoursMatch2[0]:
    if contoursMatch1[1]<=contoursMatch2[1]:
    	if contoursMatch1[2]<=contoursMatch2[2]:
            	check += 1

    if(check_old == check):
        cv2.imshow(str(i)+".scene1", scene_out)
        cv2.imshow(str(i)+".output1", output1_center)
        cv2.imshow(str(i)+".scene2", scene_out2)

    i += 1

result = float(check/i)*100

print("\n" + str(result) + "% der Tests sind erfolgreich")

cv2.waitKey(0)
cv2.destroyAllWindows()
