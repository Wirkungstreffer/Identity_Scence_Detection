
pic0 = invert_image(img[10])
pic1 = invert_image(img[15])

pic_res = pic0 + pic1
pic_res = invert_image(pic_res)


pic_flip = add_images(pic_flip, pic_res)


# Translation
rows, cols = pic_res.shape

x_translate = 100
y_translate = 50

M = np.float32([[1,0,x_translate], [0,1,y_translate]])
pic_res = cv2.warpAffine(pic_res, M, (cols,rows), borderMode = cv2.BORDER_REPLICATE)


print()
print("Verschiebungsmatrix: ")
print(M)


# cv2.imshow('Image res - translate', pic_res)
# cv2.imshow('Image res - rotate', pic_res)


# pic_flip = cv2.flip(pic_res, 0)
# cv2.imshow('flip', pic_flip)

print()
print("Random number between 0 and 10: ")
print(randint(0, 10))










contour_size = 0

for i, contour in enumerate(contours):
	contour_size += len(contour)

all_data_pts = np.empty((contour_size, 2), dtype=np.float64)

k = 0

for i, contour in enumerate(contours):

	sz = len(contour)
	data_pts = np.empty((sz, 2), dtype=np.float64)

	for i in range(data_pts.shape[0]):
		data_pts[i,0] = contour[i,0,0]
		data_pts[i,1] = contour[i,0,1]

		all_data_pts[i+k,0] = contour[i,0,0]
		all_data_pts[i+k,1] = contour[i,0,1]

	k += len(contour)

#import sys
#np.set_printoptions(threshold=sys.maxsize)
#print("{}".format(all_data_pts))

# Perform PCA analysis
mean = np.empty((0))
mean, eigenvectors, eigenvalues = cv2.PCACompute2(all_data_pts, mean)
# Store the center of the object
cntr = (int(mean[0,0]), int(mean[0,1]))

cv2.circle(scene, cntr, 4, (0, 255, 0), 1)

# p1 = x1, y1
# p2 = x2, y2
p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])

drawAxis(scene, cntr, p1, (0, 255, 0), 1)
drawAxis(scene, cntr, p2, (0, 255, 0), 1)

print()
print("Eigenvectors: {}".format(eigenvectors))
print()
print("Eigenwerte: {}".format(eigenvalues))

angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians




		print(contour[i,0,0])
		print(all_data_pts[i+k,0])
		print(contour[i,0,1])
		print(all_data_pts[i+k,1])



import sys

np.set_printoptions(threshold=sys.maxsize)
# print("{}".format(all_data_pts))

	area = cv2.contourArea(contour)
	# Ignore contours that are too small or too large
	if area < 1e2 or 1e5 < area:
	    continue

	area = cv2.contourArea(contour)
	# Ignore contours that are too small or too large
	if area < 1e2 or 1e5 < area:
	    continue




# mean = np.empty((0))
# mean, eigenvectors, eigenvalues = cv2.PCACompute2(scene, mean, maxComponents=4)

# print()
# print("mean: {}".format(mean))
# print("eigenvectors: {}".format(eigenvectors))
# print("eigenvalues: {}".format(eigenvalues))

# print(cv2.PCACompute2(scene, mean=None))



# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(scene)

# print(pca.components_)
# print(pca.explained_variance_)
