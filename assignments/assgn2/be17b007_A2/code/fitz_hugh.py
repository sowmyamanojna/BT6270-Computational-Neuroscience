import numpy as np
import matplotlib.pyplot as plt

# Initializing important parametres
a = 0.5
b, r = 0.1, 0.1

f = lambda v: v*(a-v)*(v-1)
dvdt = lambda v, w, I: f(v)-w+I
dwdt = lambda v, w: b*v-r*w

def euler_integrate(v, w, dt, I, ttot):
	niter = int(ttot/dt)
	vhist = []
	whist = []
	for i in range(niter):
		v, w = (v + dvdt(v, w, I)*dt), (w + dwdt(v, w)*dt)
		vhist.append(v)
		whist.append(w)

	return vhist, whist

def nullclines(I, v):
	vnc = f(v) + I
	wnc = b*v/r
	return vnc, wnc

def plot_fig(y, title="", ylim=(), x=None, xlim=()):
	plt.figure()
	plt.title(title)
	color = ["b", "r"]
	if not x:
		for i in y:
			plt.plot(i)
	else:
		for i,j in enumerate(y):
			plt.plot(x, j, color=color[i])
	if ylim: plt.ylim(ylim)
	if xlim: plt.xlim(xlim)
	plt.xlabel("Time")
	plt.ylabel("Voltage")
	plt.legend(["V(t)", "W(t)"])
	plt.grid()
	plt.show()

def plot_nullclines(I, xlim, ylim, positions):
	color = ['palegreen', 'gold', 'tomato', 'deepskyblue', 'lightsteelblue']*3
	v = np.linspace(xlim[0], xlim[1], 100)
	w = np.linspace(xlim[0], xlim[1], 100)
	
	v_mesh, w_mesh = np.meshgrid(v, w)
	v_vel = dvdt(v_mesh, w_mesh, I)
	w_vel = dwdt(v_mesh, w_mesh)

	vnc, wnc = nullclines(I, v)

	plt.figure()
	plt.plot(v, vnc, 'b')
	plt.plot(v, wnc, 'r')
	plt.legend(['v nullcline', 'w nullcline'])
	plt.ylim(ylim[0], ylim[1])
	title = "Phase Plot for I=" + str(round(I, 2))
	plt.title(title)
	plt.xlabel('v-values')
	plt.ylabel('w-vaues')

	if positions:
		for i in range(len(positions)):
			plt.streamplot(v_mesh, w_mesh, v_vel, w_vel, density=2, start_points=[positions[i]], color=color[i], integration_direction="forward", arrowsize=2)
	else:
		plt.streamplot(v_mesh, w_mesh, v_vel, w_vel, color=color[0])
	plt.grid()
	plt.show()

#####################################################
# Case 1: I_ext = 0
dt = 0.1
ttot = 100
I = 0

# Nullclines
plot_nullclines(I, (-0.5,1.5), (-0.45,0.45), [])
plot_nullclines(I, (-0.5,1.5), (-0.45,0.45), [[0,0], [0.4,0], [0.6, 0], [1, 0]])

# Subcase 1: V(0) < a and W(0) = 0
x = list(np.linspace(0,100,1000))
vhist, whist = euler_integrate(0.4, 0, dt, I, ttot)
plot_fig([vhist, whist], "I=0, Subcase 1: V(0) < a and W(0) = 0", (-0.2,0.75), x)

# Subcase 2: V(0) > a and W(0) = 0
vhist, whist = euler_integrate(0.6, 0, 0.1, I, 100)
plot_fig([vhist, whist], "I=0, Subcase 2: V(0) > a and W(0) = 0", (-0.2,0.75), x)

#####################################################
# Case 2: I_ext = 0.6
dt = 0.1
ttot = 100
I = 0.6

# Nullclines
plot_nullclines(I, (-0.5,1.5), (0.15,0.95), [])
plot_nullclines(I, (-0.5,1.5), (0.15,0.95), [[0,0], [0.4,0], [0.6, 0], [1, 0]])

# Subcase 1: V(0) < a and W(0) = 0
x = list(np.linspace(0,100,1000))
vhist, whist = euler_integrate(0.4, 0, dt, I, ttot)
plot_fig([vhist, whist], "I=0.6, Subcase 1: V(0) < a and W(0) = 0", (-0.2,1.5), x)

# Subcase 2: V(0) > a and W(0) = 0
vhist, whist = euler_integrate(0.6, 0, 0.1, I, 100)
plot_fig([vhist, whist], "I=0.6, Subcase 2: V(0) > a and W(0) = 0", (-0.2,1.5), x)

#####################################################
# Case 3: I_ext = 1
dt = 0.1
ttot = 100
I = 1

# Nullclines
plot_nullclines(I, (-0.5,1.5), (0.55,1.45), [])
plot_nullclines(I, (-0.5,1.5), (0.55,1.45), [[0,0.6], [0.4,0.6], [0.6, 0.6], [1, 0.6], [0,1.4], [0.4,1.4], [0.6, 1.4], [1, 1.4]])

# Subcase 1: V(0) < a and W(0) = 0
x = list(np.linspace(0,100,1000))
vhist, whist = euler_integrate(0.4, 0, dt, I, ttot)
plot_fig([vhist, whist], "I=1, Subcase 1: V(0) < a and W(0) = 0", (-0.2,2), x)

# Subcase 2: V(0) > a and W(0) = 0
vhist, whist = euler_integrate(0.6, 0, 0.1, I, 100)
plot_fig([vhist, whist], "I=1, Subcase 2: V(0) > a and W(0) = 0", (-0.2,2), x)

#####################################################
# Case 4: I_ext = 0.02
dt = 0.1
ttot = 100
I = 0.02

# Choosing parameters
b, r = 0.01, 0.8

# Nullclines
plot_nullclines(I, (-0.5,1.5), (-0.45,0.45), [])
plot_nullclines(I, (-0.5,1.5), (-0.45,0.45), [[0,0.4], [0.1,0.4], [-0.1,0.4], [0,-0.4], [0.4,0.4], [0.6, 0.4], [1, 0.4], [0.1,-0.4], [-0.1,-0.4], [0.4,-0.4], [0.6, -0.4], [1, -0.4]])

# Subcase 1: V(0) < a and W(0) = 0
x = list(np.linspace(0,100,1000))
vhist, whist = euler_integrate(0.4, 0, dt, I, ttot)
plot_fig([vhist, whist], "I=0.02, Subcase 1: V(0) < a and W(0) = 0", (-0.2,2), x)

# Subcase 2: V(0) > a and W(0) = 0
vhist, whist = euler_integrate(0.6, 0, 0.1, I, 100)
plot_fig([vhist, whist], "I=0.02, Subcase 2: V(0) > a and W(0) = 0", (-0.2,2), x)