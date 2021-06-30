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


def get_matchShapes(img0, img1, name0, name1, id):
	# MatchShapes
	contours_match = [0, 0, 0]
	contours_match[0] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I1, 0)
	contours_match[1] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I2, 0)
	contours_match[2] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I3, 0)

	print("\n"+id + ".ContoursMatch of "+ name0 +" and "+ name1 +": {}".format(contours_match))

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

    #get_huMoments(scene_out, "scene_out")

    output1 = rotate(scene_out, randint(0, 360))
    output1 = translate(output1, randint(-150, 150), randint(-150, 150))

    #get_huMoments(output1, "output1")

    check_old = check

    contoursMatch1 = get_matchShapes(invert_image(scene_out), invert_image(output1), "scene1", "output1", str(i+1))
    contoursMatch2 = get_matchShapes(invert_image(scene_out), invert_image(scene_out2), "scene1", "scene2", str(i+1))

    if contoursMatch1[0]<=contoursMatch2[0]:
        if contoursMatch1[1]<=contoursMatch2[1]:
            if contoursMatch1[2]<=contoursMatch2[2]:
                check += 1

    if(check_old == check):
        cv2.imshow(str(i)+".scene1", scene_out)
        cv2.imshow(str(i)+".output1", output1)
        cv2.imshow(str(i)+".scene2", scene_out2)

    i += 1

result = float(check/i)*100

print("\n" + str(result) + "% der Tests sind erfolgreich")

cv2.waitKey(0)
cv2.destroyAllWindows()
