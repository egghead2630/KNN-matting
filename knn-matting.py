#  Reference:
#		1.https://github.com/dingzeyuli/knn-matting/blob/master/src/knn_matting.m
#		2.https://github.com/dingzeyuli/knn-matting/blob/master/src/run_demo.m
#		3.https://github.com/MarcoForte/knn-matting/blob/master/knn_matting.py

#  Note:
# 		All data process is referenced from the paper itself and the implementation from github
#		I will try my best explaining each part, thanks for your patience
import numpy as np
import sklearn.neighbors
import scipy.sparse
import warnings
import matplotlib.pyplot as plt
import os
import cv2


def find_knn(img,trimap,nn,info):
	# build feature vector X[i] for each pixel
	# X[i] = [R, G, B, x, y], where x and y are spatial coordinates of the pixel
	[row, col, channel] = info
	indexes = np.arange(row * col)
		# index from [0,1,..., row * col - 1] for subsequent flatten image
	x, y = np.unravel_index(indexes,(row, col))
		# x stands the row number, while y stands col number for each pixel in the original img
	
	flatten_img = img.reshape((row * col), channel)
		# flatten image to be two-dim, first dim represent pixel index, while second for RGB channel
	
	x = x / np.sqrt(row * row + col * col)
	y = y / np.sqrt(row * row + col * col)
		# process spatial coordinates as the paper's github source code done
		# Level variable is ignored here for it equals to 1 in the github implementation
		# Perturb variable is ignored as well for keeping the k(i,j) value to remain in range [0,1]
		# After the processing, x and y are both in range [0,1]
	
	x = x.reshape(len(x),1)
	y = y.reshape(len(y),1)
		# reshape x and y from (len, ) to (len, 1) for concatenation
	spatial_coos = np.concatenate( (x,y), axis = 1 ) 
		# concatenate them as spatial_coos matrix to represent spatial coordinates
	
	feature_vecs = np.concatenate( (flatten_img, spatial_coos), axis = 1)
		# feature vector contains all pixels' [R,G,B,x,y] now
		# thus the size would be (row * col) * 5
	neighbor = sklearn.neighbors.NearestNeighbors(n_neighbors=nn,algorithm='kd_tree').fit(feature_vecs)
		# Build the kd tree from the feature vectors, as the paper's github source code
	
	knn = neighbor.kneighbors(feature_vecs)[1]
		# get the n nearest neighbor for each pixel, and of course each pixel is closest to itself
		# we have (row * col) pixels, each has n neighbors
		# thus knn has size of (row * col) * n, here n=10
	return knn, feature_vecs
	#print(flatten_img.shape)
def find_sparse_A(knn,my_lambda,feature_vecs,foreground,all_constraints,info):
	[row,col,channel] = info
	nn = knn.shape[1]
		# grab the info needed
	row_iterates = np.arange(row*col)
		# index on row would itrate from 0 to (row * col - 1), to indicate pixel index
		
	row_indexes = np.repeat(row_iterates, nn)
	col_indexes = knn.reshape((row * col * nn))
		# we would use this two indexes to go over the whole knn 
		# to calculate all pixel-neighbor distances
		# first all pixel 0's distances to its neighbors would be calculated
		# then 1,2 ... until pixel (row * col - 1)
	
	distances = feature_vecs[row_indexes] - feature_vecs[col_indexes]
	distances = abs(distances)		
	kernel_values =  1 - np.sum(distances, axis = 1) / (channel + 2)
		
		# apply kernel function: k(i,j) = 1 - || X[i] - X[j] || / C
		
		# C is element number in each feature vector
		# here C = 5 from the definition (R,G,B,x,y), that is channel(R,G,B) + spatial coors(x,y)
		
		# || X[i] - X[j] || is computed following the paper's github source code
		# 1. Do subtraction on two feature vector in the form:[R,G,B,x,y]:
		#    with each element(R,G,B,x,y) range in [0,1]	
		# 2. After the subtraction, each element would fall into range [-1,1]
		# 3. Apply abs() to bring range of each element back to [0,1]
		# 4. Sum all the element in the result vector to be the  || X[i] - X[j] ||
		# The above procedure would be applied to each (pixel,neighbor) pair to get their distances
		
		# each element in kernel_values must be >= 0
		# since we add up 5 values ranging [0,1] and divided by 5 to get the result range [0,1]
		# finally we subtract the result from 1, get a value range [0,1]
		
		# Note: kernel_values is of size (row * col * 10)
		# since it stores value for each (pixel,neighbor) pair
		# and we got (row * col) pixels, each of them has 10 neighbor in our case
		# thus we got the number (row * col * 10)
		
	A = scipy.sparse.csr_matrix( (kernel_values,(row_indexes,col_indexes)),shape = (row * col,row * col))
		# Contruct sparse matrix A, and construct as csr format for efficiently solving the linear system later
		
		# In the sparse matrix A:
		# We only have numbers on (i,n), where i is the pixel index and n is the pixel's neighbors' index
		# For each (i,n) pair, we store the corresponding kernel_values element in.
		# Note that both i and n ranging from 0 ~ (row * col - 1), and that's the reason we need a sparse matrix
		
		# We store totally  (row * col) * (row * col) elements in A comprised of 2 parts
		# 1. Every value in kernel_values, that is of size (row * col * 10)
		# 2. The remaining part of size (row * col) * (row * col - 10) is padded 0
	
	D = scipy.sparse.diags(np.ravel(all_constraints[:,:])).tocsr()
		# by the paper definition, D=diag(m), where m is binary vector of indices of all marked-up pixels
		# we mark up background and foreground pixels, and add up to be all_constraints
		# that is, all_constraints in our code serves as m in the paper
		# tocsr() for same reason above, no repeat below
		
	Delta = scipy.sparse.diags(np.ravel(A.sum(axis = 1))).tocsr()
		# by definition delta[i] is sigma k(i,j) over j, that in our code is
		# in A, each row i has 10 neighbor(j), and each (i,j) pair stores the corresponding kernel_values k(i,j)
		# we simply add up along ith row of A to get ith value for Delta[i]
		# Do this across the whole A, we have Delta[i] vector, with (row * col) elements
		# diagonalize Delta[i] to get the Delta vatrix
	
	Lap = Delta - A
		# As definition in paper, Laplacian = Delta - A
	
	H = 2 * (Lap + my_lambda * D)
		# derived from the paper's equation
		
	v = np.ravel(foreground[:,:])
		# By definition, v is a binary vector of pixel indices corresponding to user markups for a given layer
		# In our code, we mark up for two layer: foreground and background, and we choose 
		# to use foreground to serve as v to solve for alpha.
	c = (2 * my_lambda * np.transpose(v)).T
	
	return A,H,c
def solve_linear_system(H,c,info):
	# We solve Hx = c in this function
	[row,col,channel] = info
	warnings.filterwarnings('error')
			# when warning occurs, we jump to exception
			# two kinds of warning can occur:
			# 1. SparseEfficiencyWarning: this warning occurs when H is not csc or csr format, we transform
			# 		the type above to avoid this warning
			# 2. MatrixRankWarning: this warning occurs when H is singular, that we cannot solve H
	alpha = []
	try:
		alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(row, col)
				# If we execute this, H is not singular,use scipy.sparse.linalg.spsolve to get alpha
				# we constrain the each alpha value to range [0,1] 
				# and reshape the alpha to row * col to match with the img  
	except Warning:
		x = scipy.sparse.linalg.lsqr(H, c)
		alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(row, col)
				# same as the try part with only one difference
				# We use scipy.sparse.linalg.lsqr() to get the least-square solution to H
				# we get whole x but only x[0] is what we need, because after x[1] is some additional info but not answer
							
	return alpha
def knn_matting(img,trimap,my_lambda=100):	
	# lambda number same as paper: lambda = 100
	
	img = img / 255.0
	trimap = trimap / 255.0
	foreground = (trimap > 0.99).astype(int)
	background = (trimap < 0.01).astype(int)
	all_constraints = foreground + background
		# define foreground, background and all_constraints
		# all_constraints will serve as vector m to build up diagonal matrix D
	
	info = img.shape
		# process the img and trimap to let the RGB value in range [0,1]
		# info for the additional data needed
	n_neighbor = 10
		# find 10 nn as paper done
	print('Finding Knn')
	knn,feature_vecs = find_knn(img,trimap,n_neighbor,info)
	print('Building Sparse Matrix A')
	A,H,c = find_sparse_A(knn,my_lambda,feature_vecs,foreground,all_constraints,info)
	print('Solving linear system')
	alpha = solve_linear_system(H,c,info)

	return alpha
	
def composite(image_path,backgrounds,trimap_path):
	# Used to composite foreground with different background, after finding alphas
	print('------------Start Compositing-------------\n\n')
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		# read in images, cvtColor for format problem
	trimap = cv2.imread(trimap_path,cv2.IMREAD_GRAYSCALE)
		# trimap is read as grayscale
	alpha = knn_matting(img, trimap)
	# find alphas
	for background_path in backgrounds:
		back_img = cv2.imread(background_path)
		back_img = cv2.cvtColor(back_img,cv2.COLOR_BGR2RGB)
		back_img=cv2.resize(back_img,(img.shape[1],img.shape[0]))
			# resize background for compositing
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				for k in range(3):
					back_img[i][j][k] = int((alpha[i][j]) * img[i][j][k] + (1 - alpha[i][j]) * back_img[i][j][k])
						# Composite: C = aF + (1 - a)B
		dir_path = './results/' +  grab_file_name(background_path) + '/'
				
		if not os.path.exists(dir_path):
			print('No {} directory, Creating...\n'.format(dir_path))
			os.mkdir(dir_path)
				
		result = dir_path + grab_file_name(image_path) + '.png'
		print('Result would be store as {}\n'.format(result))	
		cv2.imwrite(result,cv2.cvtColor(back_img, cv2.COLOR_RGB2BGR))
		# store the result image
		
	print('\n\n-------------End Compositing-------------\n\n')
	return 1
	
def grab_file_name(path):
	# To get file's name without '.png' from a path
	path = path[::-1]
	# simply reverse it and record until first './', and reverse again
	cnt = 0
	name = ""
	while path[cnt] != '.':
		cnt += 1
	cnt += 1
	# filter the '.png'
	while path[cnt] != '/':
		name += path[cnt]
		cnt += 1
	# record file name
	name = name[::-1]
	return name
def get_file_paths(roots):
	# get all the file paths in specific folders  
	paths = []
	trimap_paths = []
	for root in roots: 
		for dirpath, dirname, file_names in os.walk(root):
			# get all the file names in a folder
			unit = []
			for name in file_names:
				# add some prefix to get the paths
				if root == "./image/":
					# img has same path as trimap
					tri_name = "./trimap/" + name
					name = root + name
					unit.append(name)
					trimap_paths.append(tri_name)
				else:
					name = root + name
					unit.append(name)
			paths.append(unit)
	paths.append(trimap_paths)
	return paths
	
def main():
	roots = ["./image/", "./background/"]
	paths = get_file_paths(roots)
	#print(paths)
	# get img, trimap and background img paths for compositing
	img_files = paths[0]
	back_files = paths[1]
	tri_files = paths[2]
	for i in range(len(img_files)):
		img = img_files[i]
		trimap = tri_files[i]
		print('Retrieving image: {}\n'.format(img))
				
		composite(img,back_files,trimap)


if __name__ == '__main__':
	if not os.path.exists('./results'):
		print('No results directory, Creating...\n')
		os.mkdir('./results')
		
		
	main()
