import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import mean_squared_error as mse

def get_data(name):
	"""
	Get the data from the files folder in numpy
	array format, reshape it into a column vector.
	"""
	s = np.loadtxt("../files/"+name+".txt", delimiter=",")
	s = s.reshape(-1,1)
	s = np.sign(s)
	# Make all positions that have 0 as 1
	s[np.where(s==0)] = 1
	return s

def visalize_image(data, title="", size=(90,100)):
	"""
	Helper function - Visualize an image
	"""
	if data.shape[1] == 1:
		data = data.reshape(size)
	plt.figure(figsize=[6,6])
	if title:
		plt.title(title)
	plt.imshow(data, cmap='gray')
	plt.show()

	return 0

def visalize_image_patch(image, patch, title="", size=(90,100)):
	"""
	Function to visualize an image and the generated 
	patch of the image, side-by-side.
	"""
	if image.shape[1] == 1:
		image = image.reshape(size)
	if patch.shape[1] == 1:
		patch = patch.reshape(size)
	
	plt.figure(figsize=[15,6])
	plt.subplot(1, 2, 1)
	plt.imshow(image, cmap='gray')
	plt.title(title+" - Full Image")
	plt.subplot(1, 2, 2)
	plt.imshow(patch, cmap='gray')
	plt.title(title+" - Patch")
	plt.show()

	return 0
	
def visalize_image_reconstruction(image, patch_before, patch_after, title="", size=(90,100)):
	"""
	Function to visualize the complete image, the 
	generated image before the particular epoch
	reconstruction and the reconstructed image after
	the epoch, side-by-side.
	"""
	if image.shape[1] == 1:
		image = image.reshape(size)
	if patch_before.shape[1] == 1:
		patch_before = patch_before.reshape(size)
	if patch_after.shape[1] == 1:
		patch_after = patch_after.reshape(size)
	
	plt.figure(figsize=[15,4])
	plt.subplot(1, 3, 1)
	plt.imshow(image, cmap='gray')
	plt.title(title+" - Full Image")
	plt.subplot(1, 3, 2)
	plt.imshow(patch_before, cmap='gray')
	plt.title(title+" - Reconstruction Before")
	plt.subplot(1, 3, 3)
	plt.imshow(patch_after, cmap='gray')
	plt.title(title+" - Reconstruction After")
	plt.show()

	return 0

def get_patches(data, size=(90,100)):
	"""
	Function to generate the patches.
	The patches generated are random in nature
	and the maximum height and width is specified
	through the min_x, min_y, max_x, max_y variables
	"""
	data = data.reshape(size)
	min_x = 20
	min_y = 15
	max_x = 45
	max_y = 45
	x = random.randint(0, size[0]-max_x-1)
	y = random.randint(0, size[1]-max_y-1)
	dist_x = random.randint(min_x, max_x)
	dist_y = random.randint(min_y, max_y)
	
	patch = np.zeros(size)
	patch[x:x+dist_x, y:y+dist_y] = data[x:x+dist_x, y:y+dist_y]
	
	patch = patch.reshape(-1,1)
	return patch

def plot_rmse(rmse, name, title=""):
	"""
	Function to plot the RMSE plot for each of the 
	image sets - ball, mona and cat
	"""
	for i in range(rmse.shape[1]):
		plt.figure()
		plt.plot(rmse[:,i])
		if title:
			plt.title(title + names[i])
		plt.grid(True)
		plt.xlabel("Iterations")
		plt.ylabel("RMSE")
		plt.show()

	return 0
	
def discrete_hopfield(S, W, V_new, max_epochs, flag=0, every=10, size=(90,100)):
	"""
	Discrete Hopfield Network generation
	"""
	rmse = []
	ims = []
	
	for i in tqdm(range(max_epochs), desc="Training Discrete Hopfield Network"):
		V = V_new.copy()
		V_new = np.sign(W @ V)
		
		if flag and i%every==0:
			for j in range(V_new.shape[1]):
				visalize_image_reconstruction(S[:,j].reshape(-1,1), V[:,j].reshape(-1,1), V_new[:,j].reshape(-1,1), title="Epochs:" + str(i) + " " + names[j])

		rmse_new = []
		ims_new = []
		for j in range(V.shape[1]):
			rmse_new.append(mse(S[:,j], V[:,j]))
			ims_new.append([plt.imshow(V[:,j].reshape(size), cmap="gray")])
		rmse.append(rmse_new)
		ims.append(ims_new)

	V = V_new.copy()
	rmse_new = []
	for j in range(V.shape[1]):
		rmse_new.append(mse(S[:,j], V[:,j]))
	rmse.append(rmse_new)
	
	return rmse, V, ims

def continuous_hopfield(S, W, V_new, max_epochs, lmda=10, dt=0.01, flag=0, every=10, size=(90,100)):
	"""
	Continuous Hopfield Network generation
	"""
	rmse = []
	V_hist = []
	
	U_new = V_new.copy()
	for i in tqdm(range(max_epochs), desc="Training Continuous Hopfield Network"):
		U = U_new.copy()
		V = V_new.copy()
		
		U_new = U + (-U + W @ V)*dt
		V_new = np.tanh(lmda*U_new)
		
		if flag and i%every==0:
			for j in range(V_new.shape[1]):
				visalize_image_reconstruction(S[:,j].reshape(-1,1), V[:,j].reshape(-1,1), V_new[:,j].reshape(-1,1), title="Epochs:" + str(i) + " " + names[j])

		rmse_new = []
		for j in range(V.shape[1]):
			rmse_new.append(mse(S[:,j], V[:,j]))
		rmse.append(rmse_new)
		
		V_hist.append(V)

	V = V_new.copy()
	rmse_new = []
	for j in range(V.shape[1]):
		rmse_new.append(mse(S[:,j], V[:,j]))
	rmse.append(rmse_new)
	V_hist.append(V)
	
	return rmse, V, V_hist

# def get_animation(fig, ims, name, title=""):
# 	"""
# 	Function to save a list of images as an animation
# 	"""
# 	ball = [i[0] for i in ims]
# 	mona = [i[1] for i in ims]
# 	cat = [i[2] for i in ims]

# 	for j in range(len(name)):
# 		ani = animation.ArtistAnimation(fig, [i[j] for i in ims], interval=50, blit=True, repeat_delay=1000)
# 		ani.save(title+"_"+name[j]+".mp4", dpi=600)

# 	return 0

def damage_weights(W, p):
	"""
	Function to damage the weights based on the
	fraction of damage specified
	"""
	W_damaged = W.copy()
	N = W.shape[0]
	pos = np.random.randint(0, N*N-1, size=(int(N*N*p),1))
	W_damaged = W_damaged.reshape(-1,1)
	W_damaged[pos] = 0
	
	return W_damaged.reshape(N,N)

def save_figures(V_hist, names, p=0, title=""):
	"""
	Function to save all the images generated
	"""
	n = V_hist[0].shape[1]
	path = os.getcwd()
	for j in range(n):
		try:
			os.chdir(names[j])
		except:
			os.mkdir(names[j])
			os.chdir(names[j])
		for i in tqdm(range(len(V_hist)), desc="Saving Figures; Part "+str(j)):
			plt.imshow(V_hist[i][:,j].reshape(90,100), cmap="gray")
			plt.title(f"Weight Damage {p}% Epochs: "+str(i))
			plt.savefig(title + str(i) + ".png")
		os.chdir(path)
	return 0


#################################################
#################################################
#################################################
# Get initial data
print("Loading all images...", end=" ")
ball = get_data('ball')
mona = get_data('mona')
cat = get_data('cat')
print("Done!")

#################################################
# Get patches of the data and visualize them
print("Getting patches of all images...", end="")
ball_patch = get_patches(ball)
mona_patch = get_patches(mona)
cat_patch = get_patches(cat)

visalize_image_patch(ball, ball_patch, "Ball")
visalize_image_patch(mona, mona_patch, "Mona")
visalize_image_patch(cat, cat_patch, "Cat")
print(" Done!")

#################################################
# Append and store them in the matrix S, V
S = np.c_[ball, mona, cat]
V = np.c_[ball_patch, mona_patch, cat_patch]

V_init = V.copy()
V_new = V.copy()

# Calculate weights
print("Calculating weights...", end=" ")
N = S.shape[0]
W = (1/N)*(S @ S.T)
print("Done!")

# Save order of images
names = ["Ball", "Mona", "Cat"]

#################################################
# Discrete Hopfield Network
print("="*50)
max_epochs = 10
rmse, V_final, ims = discrete_hopfield(S, W, V_new, max_epochs, flag=0)

rmse = np.array(rmse)
plt.close()
plot_rmse(rmse, names, title="Discrete Hopfield; RMSE after training on ")

#################################################
# Continuous Hopfield Network
save_images_flag = 0

print("="*50)
max_epochs = 40
lmda = 20
dt = 0.01
rmse, V_final, V_hist_0 = continuous_hopfield(S, W, V_new, max_epochs, lmda=lmda, dt=dt, flag=0, every=10)

rmse = np.array(rmse)
plt.close()
plot_rmse(rmse, names, title="Continuous Hopfield; RMSE after training on ")
if save_images_flag:
	save_figures(V_hist_0, names, p=0, title=f"chn_0_")

#################################################
#################################################
print("="*50)
# Weight Damage
print("Training with Damaged Weights")
## 25% weight damage
print("25% Weight Damage")
max_epochs = 40
p = 0.25
rmse, V_final, V_hist_25 = continuous_hopfield(S, damage_weights(W, p), V_new, max_epochs, lmda=lmda, dt=dt, flag=0, every=10)

rmse = np.array(rmse)
plt.close()
plot_rmse(rmse, names, title=f"Continuous Hopfield - {p*100}% damage; RMSE after training on ")
if save_images_flag:
	save_figures(V_hist_25, names, p=25, title=f"chn_25_")

#################################################
## 50% weight damage
print("="*50)
print("50% Weight Damage")
p = 0.50
max_epochs = 50
rmse, V_final, V_hist_50 = continuous_hopfield(S, damage_weights(W, p), V_new, max_epochs, lmda=lmda, dt=dt, flag=0, every=10)

rmse = np.array(rmse)
plt.close()
plot_rmse(rmse, names, title=f"Continuous Hopfield - {p*100}% damage; RMSE after training on ")
if save_images_flag:
	save_figures(V_hist_50, names, p=50, title=f"chn_50_")

#################################################
## 80% weight damage
print("="*50)
print("80% Weight Damage")
p = 0.80
max_epochs = 70
rmse, V_final, V_hist_80 = continuous_hopfield(S, damage_weights(W, p), V_new, max_epochs, lmda=lmda, dt=dt, flag=0, every=10)

rmse = np.array(rmse)
plt.close()
plot_rmse(rmse, names, title=f"Continuous Hopfield - {p*100}% damage; RMSE after training on ")
if save_images_flag:
	save_figures(V_hist_80, names, p=80, title=f"chn_80_")
