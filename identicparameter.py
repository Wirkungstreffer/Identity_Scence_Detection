from random import randint
import cv2
import math
import numpy as np

print("x"*50 + "\nHu-Moment-Test\n" + "x"*50 + "\n")

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
	contours_match = [0, 0, 0]
	contours_match[0] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I1, 0)
	contours_match[1] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I2, 0)
	contours_match[2] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I3, 0)

	print("\nContoursMatch of "+ name0 +" and "+ name1 +": {}".format(contours_match))

	return contours_match

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

#Generiere 1000 Szenen
sceneList = []
j = 0

kernel = np.ones((5,5),np.uint8)

while (j<1000):
	sceneList.append(generateScene())
	sceneList[j] = cv2.erode(sceneList[j],kernel,iterations = 1)
	sceneList[j] = fillContour(sceneList[j])
	j += 1
	print(str(j)+" Szenen generiert")
	
#Generiere Orginalszene

scene = generateScene()

scene_out = cv2.erode(scene,kernel,iterations = 1)
scene_out = fillContour(scene_out)

output1 = rotate(scene_out, randint(0, 360))
output1 = translate(output1, randint(-150, 150), randint(-150, 150))

contoursMatch1 = get_matchShapes(invert_image(scene_out), invert_image(output1), "scene1", "output1")


cv2.imshow("scene1", scene_out)
cv2.imshow("output1", output1)

#Vergleich und Speicherung der 10 kleinsten Match-Werte

minMatchValueList = []
contoursMatchList = []

i = 0

while(i<1000):

	contoursMatch2 = get_matchShapes(invert_image(output1), invert_image(sceneList[i]), "Output1", "scene2")

	contoursMatchList.append(contoursMatch2)
	
	i += 1
	
contoursMatchList.sort()

k = 0

while(k<10):
	
	minMatchValueList.append(contoursMatchList[k])
	
	i = 0
	while(i<1000):
		contoursMatch2 = get_matchShapes(invert_image(output1), invert_image(sceneList[i]), "Output1", "scene2")
		if (minMatchValueList[k][0] == contoursMatch2[0]):
			if (minMatchValueList[k][1] == contoursMatch2[1]):
				if (minMatchValueList[k][2] == contoursMatch2[2]):
					cv2.imshow(str(k)+".scene2", sceneList[i])
		i += 1
	
	k += 1

j = 0
while(j<10):
	print(str(j)+". " + str(minMatchValueList[j]))
	j += 1
print(str(contoursMatch1))

cv2.waitKey(0)
cv2.destroyAllWindows()
