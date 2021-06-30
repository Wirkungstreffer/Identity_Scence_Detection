# PCA
# mean, eigenVectors = cv2.PCACompute(pic_res, mean=None, maxComponents=2)

def getOrientation(contour, img):

	sz = len(contour)
	data_pts = np.empty((sz, 2), dtype=np.float64)

	for i in range(data_pts.shape[0]):
	    data_pts[i,0] = contour[i,0,0]
	    data_pts[i,1] = contour[i,0,1]

	import sys
	np.set_printoptions(threshold=sys.maxsize)
	# print(data_pts)
	# Perform PCA analysis
	mean = np.empty((0))
	mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
	# Store the center of the object
	cntr = (int(mean[0,0]), int(mean[0,1]))

	cv2.circle(img, cntr, 4, (0, 255, 0), 1)

	# p1 = x1, y1
	# p2 = x2, y2
	p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
	p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])

	drawAxis(img, cntr, p1, (0, 255, 0), 1)
	drawAxis(img, cntr, p2, (0, 255, 0), 1)

	print()
	print("Eigenvectors: {}".format(eigenvectors))
	print()
	print("Eigenwerte: {}".format(eigenvalues))

	angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

	return angle

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



# PCA for every contour
for i, c in enumerate(contours):
	# Calculate the area of each contour
	area = cv2.contourArea(c)
	# Ignore contours that are too small or too large
	if area < 1e2 or 1e5 < area:
	    continue
	# Draw each contour only for visualisation purposes
	# cv2.drawContours(pic_res, contours, i, (0, 255, 0), 1)
	# Find the orientation of each shape

	# getOrientation(c, scene)
