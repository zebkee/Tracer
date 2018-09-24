#Determines the optical efficiency of a heliostat field -> receiver pairing at at different sun positions (a grid of azimuth, elevation pairs)
#Optical efficiency is the result of cosine, shading, reflectivity, blocking, spillage losses.

#Heliostat generator is used from Ye's code

#Python Packages
import numpy as N
from numpy import r_
from scipy.constants import degree
import matplotlib.pyplot as plt
import copy
import timeit
#Tracer Packages
from tracer.ray_bundle import RayBundle
from tracer.sources import pillbox_sunshape_directions
from tracer.sources import solar_disk_bundle
from tracer.sources import buie_sunshape
from tracer.assembly import Assembly
from tracer.spatial_geometry import roty, rotx, rotz, rotation_to_z
from tracer.models.one_sided_mirror import rect_one_sided_mirror, rect_para_one_sided_mirror
from tracer.tracer_engine import TracerEngine
from tracer.optics_callables import *
#Tracer Models
from tracer.models.heliostat_field import HeliostatGenerator, KnownField, SinglePointAiming, TiltRoll, AzElTrackings
#Coin3D renderer
from tracer.CoIn_rendering.rendering import *

#Receiver Models
from tracer.flat_surface import RectPlateGM
from tracer.surface import Surface
from tracer.spatial_geometry import general_axis_rotation
from tracer.object import AssembledObject

def search_final(H_mean,H_sigma):
	"""Searches the H_mean array for the largest value and then finds its associated standard deviation value"""
	rows = len(H_mean)
	cols = len(H_mean[0])

	index = N.argmax(H_mean)
	rowi = index//cols
	coli = index%cols
	
	maxi = H_mean[rowi][coli]
	sigma = H_sigma[rowi][coli]

	if N.amax(H_mean) != maxi:
		print("Warning, index search has an error!")
	return maxi,sigma

def solar_angles(declination,HRA,latitude):
	degree = 2*N.arcsin(1.0)/180.0
	elevation = N.arcsin(N.sin(degree*declination)*N.sin(degree*latitude)+N.cos(degree*declination)*N.cos(degree*latitude)*N.cos(degree*HRA))/degree
	ARC = (N.sin(degree*declination)*N.cos(degree*latitude)-N.cos(degree*declination)*N.sin(degree*latitude)*N.cos(degree*HRA))/N.cos(degree*elevation)
	if ARC < -1.0:
		print("Argument of arccos is less than -1.0")
		ARC = -1.0
	if ARC > 1.0:
		print("Argument of arccos is more than 1.0")
		ARC = 1.0
	azimuth = N.arccos(ARC)/degree
	if HRA > 0.0:
		azimuth = 360.0 - azimuth
	return azimuth, elevation


def solar_vector(azimuth, zenith):
	"""
	Calculate the solar vector using elevation and azimuth.

	Arguments:
	azimuth - the sun's azimuth, in radians from North, clockwise.
	elevation - angle created between the solar vector and the XY Plane

	Returns: a 3-component 1D array with the solar vector.
	"""
	elevation = N.pi/2.0 - zenith
	sun_z = N.sin(elevation)
	sun_xy = N.r_[N.sin(azimuth), N.cos(azimuth)] # unit vector, can't combine with z
	sun_vec = N.r_[sun_xy*N.sqrt(1 - sun_z**2), sun_z]
	return sun_vec

def rotation_matrix(x_ang,y_ang,z_ang):
    """
    Returns the rotation matrix of a rotation about each coordinate axes done one after another. Input in Radians.
    """
    #remove my
    mx = N.matrix(general_axis_rotation(r_[1,0,0],x_ang))
    my = N.matrix(general_axis_rotation(r_[0,1,0],y_ang))
    mz = N.matrix(general_axis_rotation(r_[0,0,1],z_ang))
    M1 = my*mx
    M2 = mz*M1
    return M2

def two_sided_receiver(width, height, absorptivity=1., location=None):
	"""
	Constructs a receiver centred at a location and parallel to the xz plane
	
	Arguments:
	width - the extent along the x axis in the local frame.
	height - the extent along the y axis in the local frame.
	absorptivity - the ratio of energy incident on the reflective side that's
		not reflected back.
	
	Returns:
	front - the receiving surface
	obj - the AssembledObject containing both surfaces
	"""
	front = Surface(RectPlateGM( height, width),ReflectiveReceiver(absorptivity),location,rotation=rotation_matrix(N.pi/2.0,0.0,0.0))
	obj = AssembledObject(surfs=[front])
	return front, obj

def tableline(col1, data): #this is for the modelica table
	"""converts col1=1.0 and data=[4.0,5.0,6.0] to a string= 1.0 4.0 5.0 6.0 /n"""
	line = str(col1)+" "
	for point in data:
		line += str(point)+" "
	line += "\n"
	return line

def save_hist(H_norm,extent,index=1,filename="histogram",dpi=500):
	"""Saves a histogram and the lines which separate each surface"""
	plt.rcParams['axes.linewidth'] = 0.5 #set the value globally
	img = plt.imshow(1e-3*H_norm,extent = extent,interpolation='nearest',cmap="jet")
	cbar = plt.colorbar(orientation='horizontal',label="Flux (kW/m2)")

	# Display the vertical lines
	newstring = filename+"/"+str(index)+".png"
	plt.xlabel("width (m)")
	plt.ylabel("height (m)")
	plt.title(newstring)
	plt.savefig(open(newstring, 'w'),dpi=dpi)
	plt.close('all')
	return

def stats(A,Q,x,N):
	"""Takes the cumulative mean, new data value and number of samples, and returns the new mean, Q and standard error"""
	#print("Sample Size",n)
	A_new = ((float(N)-1.0)*A+x)/float(N)
	Q_new = Q + (x-A)*(x-A_new)
	#sepct = N.nan_to_num(100*sigma/(mean*(n**0.5)))
	sd = (Q_new/(float(N)-1.0))**0.5
	se = sd/((float(N))**0.5)
	return A_new,Q_new,se

def search_final(H_mean,H_sigma):
	"""Searches the H_mean array for the largest value and then finds its associated standard deviation value"""
	rows = len(H_mean)
	cols = len(H_mean[0])

	index = N.argmax(H_mean)
	rowi = index//cols
	coli = index%cols
	
	maxi = H_mean[rowi][coli]
	sigma = H_sigma[rowi][coli]

	if N.amax(H_mean) != maxi:
		print("Warning, index search has an error!")
	return maxi,sigma

def stringed(array):
	"""Converts an array of values to a string of comma separated values"""
	S = ""
	for item in array:
		S += str(item)
		S += ","
	return S

###Trace Scene
class TowerSceneZeb():
	""" Square Blackbody receiver on the xz plane with set area. Custom heliotat field"""
	# rec_area = area of the receiver (assumed square)
	# rec_centre = coordinates of the centre of the receiver (global)
	# heliostat is a csv file of coordinates (Example: sandia_hstat_coordinates.csv)
	#   x,y,z,focal_length
	# azimuth and elevation angles are in radians
	def __init__(self,recv_area,recv_centre,heliostat,azimuth = 0.0,elevation = 0.0,helio_w=1.85,helio_h=2.44,helio_abs=0.10,helio_sigmaxy=1.5e-3,helio_tracking="TiltRoll"):
		self.recv_surf, self.recv_obj = two_sided_receiver(recv_area[0],recv_area[1],location=recv_centre,absorptivity=1.00)	
		#Sun angles in radians	
		self.azimuth = azimuth
		self.zenith = N.pi/2.0 - elevation
		self.elevation = elevation
		# add the heliostat coordinates
		self.pos = N.loadtxt(heliostat, delimiter=",")[:,0:3] #These are positions x,y,z (m)
		self.helio_focal = N.loadtxt(heliostat, delimiter=",")[:,3] #These are focal lengths (m)
		self.layout = N.loadtxt(heliostat, delimiter=",")[:,0:4] #Layout of the field
		self.helio_area = float(len(self.pos))*helio_w*helio_h #These is the total heliostat area (m^2)
		print("heliostat area",self.helio_area,"m2")
		self.helio_w = helio_w #This is heliostat width
		self.helio_h = helio_h #This is heliostat height
		self.helio_abs = helio_abs #This is heliostat absorptivity
		self.helio_sigmaxy = helio_sigmaxy #This is surface slope error of mirrors
		self.helio_tracking = helio_tracking #This is tracking strategy "AzEl" or "TiltRoll"
		self.heliostatcsv = heliostat

		self.recv_centre = recv_centre
		self.dz = recv_centre[2] #The height of the receiver centre

		# Obtain x,y centre and radius of raybundle source
		self.field_centroid = sum(self.pos)/float(len(self.pos))
		self.field_radius = (((max(self.pos[:,0])-min(self.pos[:,0]))**2.0+(max(self.pos[:,1])-min(self.pos[:,1]))**2.0)**0.5)*0.5
		# generate the entire plant now
		self.gen_plant()
		# creates an attribute which shows number of rays used, start at zero

	def gen_rays(self,rays):
		sun_vec = solar_vector(self.azimuth, self.zenith)
		source_centre = N.array([[self.field_centroid[0]+200.0*sun_vec[0]],[self.field_centroid[1]+200.*sun_vec[1]],[self.field_centroid[2]+200.0*sun_vec[2]]])
		#print(self.field_centroid + (500.0*sun_vec))

		#print(sun_vec_array)
		self.DNI = 1000.0
		self.rays = rays
		#self.raybundle = buie_sunshape(self.rays, source_centre, -1.0*sun_vec, 165.0, 0.0225, flux=self.DNI)
		self.raybundle = solar_disk_bundle(self.rays, source_centre, -1.0*sun_vec, self.field_radius, 4.65e-3, flux=self.DNI)
		#print(self.DNI, sum(self.raybundle.get_energy()))
		return rays

	def gen_plant(self):
		"""Generates the entire plant"""
		# set heliostat field characteristics: 6.09m*6.09m, abs = 0.04, aim_location_xyz =60
		self.field = HeliostatGenerator(self.helio_w,self.helio_h,self.helio_abs,self.helio_sigmaxy,slope='normal',curved=True,pos=self.pos,foc=self.helio_focal)
		self.field(KnownField(self.heliostatcsv,self.pos,N.array([self.helio_focal]).T)) #field(layout)
		heliostats=Assembly(objects=self.field._heliostats)
		aiming=SinglePointAiming(self.pos, self.recv_centre, False)
		if self.helio_tracking =='TiltRoll':
			tracking=TiltRoll(solar_vector(self.azimuth, self.zenith),False)	
		elif self.helio_tracking =='AzEl':
			tracking=AzElTrackings(solar_vector(self.azimuth, self.zenith),False) 
		tracking.aim_to_sun(-self.pos, aiming.aiming_points, False)
		tracking(self.field)

		self.plant = Assembly(objects = [self.recv_obj], subassemblies=[heliostats])

		##### Calculate Total Recv area #####
		areacount=0.
		corners = self.recv_surf.mesh(0) #corners is an array of all corners of the plate
		# BLC is bottom left corner "origin" of the histogram plot
		# BRC is the bottom right corner "x-axis" used for vector u
		# TLC is the top right corner "y-axis" used for vector v
		BLC = N.array([corners[0][1][1],corners[1][1][1],corners[2][1][1]])
		BRC = N.array([corners[0][0][1],corners[1][0][1],corners[2][0][1]])
		TLC = N.array([corners[0][1][0],corners[1][1][0],corners[2][1][0]])
		# Get vectors u and v in array form of array([x,y,z])
		u = BRC - BLC
		v = TLC - BLC
		# Get width(magnitude of u) and height(magnitude of v) in float form
		w = (sum(u**2))**0.5
		h = (sum(v**2))**0.5
		areacount += w*h
		self.recv_area = areacount
		print("Reciver area",self.recv_area)

	#def aim_field(self):
		#"""Aims the field to the sun?"""
		#self.sun_vec = solar_vector(self.azimuth,self.zenith)
		#aiming = SinglePointAiming(self.pos, self.recv_centre, False)
		#if self.helio_tracking == "TiltRoll":
			#tracking = TiltRoll(self.sun_vec,False)
		#else:
			#tracking = AzElTrackings(self.sun_vec,False)
		#tracking.aim_to_sun(-self.pos, aiming.aiming_points, False)
		#tracking(self.field.get_objects())
		#self.field.aim_to_sun(self.azimuth, self.elevation)

	def trace(self, iters = 10000, minE = 1e-9, render = False,bins=20):
		"""Commences raytracing using (rph) number of rays per heliostat, for a maximum of 
		   (iters) iterations, discarding rays with energy less than (minE). If render is
		   True, a 3D scene will be displayed which would need to be closed to proceed."""
		# Get the solar vector using azimuth and elevation

		raytime0 = timeit.default_timer()
		# Perform the raytracing
		e = TracerEngine(self.plant)
		e.ray_tracer(self.raybundle, iters, minE, tree=True)
		e.minener = minE

		raytime1 = timeit.default_timer()
		print("RayTime",raytime1-raytime0)
		#power_per_ray = self.power_per_ray
		self.power_per_ray = self.DNI/self.rays
		power_per_ray = self.power_per_ray

		# Optional rendering
		if render == True:
			trace_scene = Renderer(e)
			trace_scene.show_rays()


		#pe0 is the array of energies of rays in bundle 0 that have children
		i_list = e.tree._bunds[1].get_parents()
		e_list = e.tree._bunds[0].get_energy()

		pe0 = N.array([])
		for i in i_list:
			pe0=N.append(pe0,e_list[i])

		#pe1 is the array of energies of rays in bundle 1 that have children
		i_list = e.tree._bunds[2].get_parents()
		e_list = e.tree._bunds[1].get_energy()

		pe1 = N.array([])
		for i in i_list:
			pe1=N.append(pe1,e_list[i])

		#pz0 is the array of z-vertices of rays in bundle 0 that have children
		i_list = e.tree._bunds[1].get_parents()
		z_list = e.tree._bunds[0].get_vertices()[2]

		pz0 = N.array([])
		for i in i_list:
			pz0=N.append(pz0,z_list[i])

		#az1 is the array of z-vertices of rays in bundle 1 that have children
		i_list = e.tree._bunds[2].get_parents()
		z_list = e.tree._bunds[1].get_vertices()[2]

		pz1 = N.array([])
		for i in i_list:
			pz1=N.append(pz1,z_list[i])

		#Helio_0 initial rays incident on heliostat
##############ASSUMING PERFECT REFLECTORS, HELIO1 IS ALSO RAYS COMING OUT OF THE HELIOSTAT INITIALLY#######(NOT CORRECTED FOR BLOCK)
		#azh1 = array of z-values less than half of recv height
		azh1 = e.tree._bunds[1].get_vertices()[2]<self.dz/2.0 
		#array of parent energies of those rays is pe0
		self.helio_0 = float(sum(azh1*pe0))#*power_per_ray

		#Helio_1 is all rays coming out of a heliostat
		#ape1 is array of all energies in bundle 1
		ape1 = e.tree._bunds[1].get_energy()
		self.helio_1 = float(sum(ape1*azh1))#*power_per_ray

		#Recv_0 initial rays incident on receiver
		#azr1 = array of z-values greater than half of recv height
		azr1 = e.tree._bunds[1].get_vertices()[2]>self.dz/2.0
		self.recv_0 = float(sum(azr1*pe0))#*power_per_ray
		
		#Helio_b rays out of heliostat that are blocked by other mirrors
		#azh2 = array of z-values less than half of recv height of bundle 2
		azh2 = e.tree._bunds[2].get_vertices()[2]<self.dz/2.0
		#pz1 = array of z-values of bundle 1 that have children
		pzh1 = pz1<self.dz/2.0
		#pe1 = array of energies of bundle 1 that have children
		self.helio_b = float(sum(azh2*pzh1*pe1))#*power_per_ray
		
		#Helio_2 is effectively what comes out of the heliostats
		self.helio_2 = self.helio_1-self.helio_b #This is effective power out of heliostat in kW

		#Recv_2 is rays hitting receiver from heliostat
		azr2 = e.tree._bunds[2].get_vertices()[2]>self.dz/2.0
		pzh1 = pz1<self.dz/2.0
		self.recv_1 = float(sum(azr2*pzh1*pe1))#*power_per_ray


		totalabs = 0
		#energy locations
		front = 0
		back = 0
		for surface in (self.plant.get_local_objects()[0]).get_surfaces():
			energy, pts = surface.get_optics_manager().get_all_hits()
			totalabs += sum(energy)
			#if surface.iden == "front":
				#front += sum(energy)

			#if surface.iden == "back": 
				#back += sum(energy)
		#Cosine efficiency calculations
		sun_vec = solar_vector(self.azimuth, self.zenith)
		tower_vec = -1.0*self.pos 
		tower_vec += self.recv_centre
		tower_vec /= N.sqrt(N.sum(tower_vec**2, axis=1)[:,None])
		hstat = sun_vec + tower_vec
		hstat /= N.sqrt(N.sum(hstat**2, axis=1)[:,None])
		self.cos_efficiency = sum(N.dot(sun_vec,(hstat).T))/float(len(hstat))


		#Intermediate Calculation
		self.cosshadeeff = (self.helio_0)/(self.DNI*self.helio_area)

		#Results
		self.p_recvabs = totalabs #Total power absrorbed by receiver

		self.coseff = self.cos_efficiency
		self.shadeeff = self.cosshadeeff/self.coseff
		self.refleff = (self.helio_1)/(self.helio_0)
		self.blockeff = (self.helio_1-self.helio_b)/(self.helio_1)
		self.spilleff = (self.recv_1)/(self.helio_2)
		self.abseff = (self.p_recvabs-self.recv_0)/(self.recv_1)
		self.opteff = (self.p_recvabs-self.recv_0)/(self.DNI*self.helio_area) #OpticalEfficiency
		self.hist_out = self.hist_flux(bins)
		self.Square_spilleff = (self.SquareEnergy - self.recv_0)/(self.helio_2) #provided absorptivity of recv is 1.0
		self.Square_opteff = (self.SquareEnergy - self.recv_0)/(self.DNI*self.helio_area) #ditto

	def hist_flux(self, no_of_bins=1000):
		"""Returns a combined histogram of all critical surfaces and relevant data"""
		# H is the histogram array
		# boundlist is a list of plate boundaries given in x coordinates
		# extent is a list of [xmin,xmax,ymin,ymax] values
		# binarea is the area of each bin. Used to estimate flux concentration

		# Define empty elements
		X_offset = 0	# Used to shift values to the right for each subsequent surface
		all_X = []	# List of all x-coordinates
		all_Y = []	# List of all y-coordinates
		all_E = []	# List of all energy values
		boundlist = [0]	# List of plate boundaries, starts with x=0

		#print("length here"+str(len((self.plant.get_local_objects()[0]).get_surfaces())))

		#for plate in self.crit_ls:	#For each surface within the list of critical surfs
		#crit_length = len(self.crit_ls)
		count = 0

		#surface = (self.plant.get_local_objects()[0]).get_surfaces()[count]
		#print(surface)

		energy, pts = self.recv_surf.get_optics_manager().get_all_hits()
		corners = self.recv_surf.mesh(0) #corners is an array of all corners of the plate
		# BLC is bottom left corner "origin" of the histogram plot
		# BRC is the bottom right corner "x-axis" used for vector u
		# TLC is the top right corner "y-axis" used for vector v
		BLC = N.array([corners[0][1][1],corners[1][1][1],corners[2][1][1]])
		BRC = N.array([corners[0][0][1],corners[1][0][1],corners[2][0][1]])
		TLC = N.array([corners[0][1][0],corners[1][1][0],corners[2][1][0]])
		# Get vectors u and v in array form of array([x,y,z])
		u = BRC - BLC
		v = TLC - BLC
		# Get width(magnitude of u) and height(magnitude of v) in float form
		w = (sum(u**2))**0.5
		h = (sum(v**2))**0.5
		# Get unit vectors of u and v in form of array([x,y,z])
		u_hat = u/w
		v_hat = v/h
		# Local x-position determined using dot product of each point with direction
		# Returns a list of local x and y coordinates
		origin = N.array([[BLC[0]],[BLC[1]],[BLC[2]]])
		local_X = list((N.array(N.matrix(u_hat)*N.matrix(pts-origin))+X_offset)[0])
		#local_Y = list((N.array(N.matrix(v_hat)*N.matrix(pts-origin)))[0])
		local_Y = list((((N.array(N.matrix(v_hat)*N.matrix(pts-origin)))[0])*-1)+h)
		# Adds to the lists
		all_X += local_X
		all_Y += local_Y
		all_E += list(energy)
		X_offset += w
		boundlist.append(X_offset)
		count += 1
		# Now time to build a histogram
		rngy = h
		rngx = X_offset
		bins = [no_of_bins,no_of_bins]
		H,ybins,xbins = N.histogram2d(all_Y,all_X,bins,range=([0,rngy],[0,rngx]), weights=all_E)
		
		#22/08/18 Add in the concentric squares code Note: Assumes 10mx10m SQUARE, 20*20 bins
		#self.SquareEnergy = [1,2,3,4,5,6,7,8,9,10] where the number is side length eg 4 x 4.
		L = 1 #Start with side length = 1
		absorb_list = [] #Start with empty list of all energy absorbed in that square
		while L <= 10: #Stops at 10.0m x 10.0m

			absorbed = N.sum(H[(10-L):(10+L),(10-L):(10+L)])
			absorb_list.append(absorbed)
			L += 1
		self.SquareEnergy = N.array(absorb_list)

		extent = [xbins[0],xbins[-1],ybins[0],ybins[-1]]
		binarea = 1.0*(float(h)/no_of_bins)*(float(X_offset)/int(no_of_bins*X_offset)) #this is in metres
		#print("maxH",N.amax(H))
		return [H/binarea, boundlist, extent]

class Stochastic():
	"""Determines the efficiency of the field-receiver using stochastic raytracing"""
	def __init__(self,recv_area,recv_centre,heliostat,azimuth = 0.0,elevation = 0.0,helio_w=1.85,helio_h=2.44,helio_abs=0.10,helio_sigmaxy=1.5e-3,helio_tracking="TiltRoll",bins=100,rays_per_run=100000,precision=1.0,render=False):

		self.rays_per_run = rays_per_run
		self.precision = precision
		self.render = render
		self.bins = bins

		self.recv_area = recv_area
		self.recv_centre = recv_centre
		self.heliostat = heliostat
		self.azimuth = azimuth
		self.elevation = elevation
		self.helio_w = helio_w
		self.helio_h = helio_h
		self.helio_abs = helio_abs
		self.helio_sigmaxy = helio_sigmaxy
		self.helio_tracking = helio_tracking
		self.bins = bins
		self.rays_per_run = rays_per_run
		self.precision = precision
		self.render = render

	def trace(self):

		self.iteration = 1
		self.total_rays = 0


		self.scene = TowerSceneZeb(self.recv_area,self.recv_centre,self.heliostat,self.azimuth,self.elevation,self.helio_w,self.helio_h,self.helio_abs,self.helio_sigmaxy,self.helio_tracking)
		#self.scene.aim_field()
		print("Azimuth",N.degrees(self.azimuth),"Elevation",N.degrees(self.elevation),"Iteration",self.iteration)
		self.total_rays += self.scene.gen_rays(self.rays_per_run)
		self.scene.trace(render = self.render,bins=self.bins)
		self.render = False

		self.coseff_mean = self.scene.coseff
		self.shadeeff_mean = self.scene.shadeeff
		self.refleff_mean = self.scene.refleff
		self.blockeff_mean = self.scene.blockeff
		self.spilleff_mean = self.scene.spilleff
		self.abseff_mean = self.scene.abseff
		self.opteff_mean = self.scene.opteff
		self.hist_mean = self.scene.hist_out[0]

		self.Square_opteff_mean = self.scene.Square_opteff
		self.Square_spilleff_mean = self.scene.Square_spilleff
		

		self.peakflux_mean = N.amax(self.hist_mean)
		

		self.coseff_Q = 0.0
		self.shadeeff_Q = 0.0
		self.refleff_Q = 0.0
		self.blockeff_Q = 0.0
		self.spilleff_Q = 0.0
		self.abseff_Q = 0.0
		self.opteff_Q = 0.0

		self.Square_opteff_Q = self.Square_opteff_mean*0.0
		self.Square_spilleff_Q = self.Square_spilleff_mean*0.0
		#print(self.hist_mean)
		self.hist_Q = self.hist_mean*0.0
		self.boundlist = self.scene.hist_out[1]
		self.extent = self.scene.hist_out[2]

		self.iteration += 1
		self.error_pct = 100.0

		while (self.iteration <= 5 or 1.0*self.error_pct > self.precision):
			print("Azimuth",N.degrees(self.azimuth),"Elevation",N.degrees(self.elevation),"Iteration",self.iteration)
			self.scene = TowerSceneZeb(self.recv_area,self.recv_centre,self.heliostat,self.azimuth,self.elevation,self.helio_w,self.helio_h,self.helio_abs,self.helio_sigmaxy,self.helio_tracking)
			#self.scene.aim_field()
			self.total_rays += self.scene.gen_rays(self.rays_per_run)
			self.scene.trace(render = self.render,bins=self.bins)
			self.render = False

			self.coseff_new = self.scene.coseff
			self.shadeeff_new = self.scene.shadeeff
			self.refleff_new = self.scene.refleff
			self.blockeff_new = self.scene.blockeff
			self.spilleff_new = self.scene.spilleff
			self.abseff_new = self.scene.abseff
			self.opteff_new = self.scene.opteff

			self.Square_opteff_new = self.scene.Square_opteff
			self.Square_spilleff_new = self.scene.Square_spilleff

			self.hist_new = self.scene.hist_out[0]

			self.coseff_mean,self.coseff_Q,self.coseff_error = stats(self.coseff_mean,self.coseff_Q,self.coseff_new,self.iteration)
			self.shadeeff_mean,self.shadeeff_Q,self.shadeeff_error = stats(self.shadeeff_mean,self.shadeeff_Q,self.shadeeff_new,self.iteration)
			self.refleff_mean,self.refleff_Q,self.refleff_error = stats(self.refleff_mean,self.refleff_Q,self.refleff_new,self.iteration)
			self.blockeff_mean,self.blockeff_Q,self.blockeff_error = stats(self.blockeff_mean,self.blockeff_Q,self.blockeff_new,self.iteration)
			self.spilleff_mean,self.spilleff_Q,self.spilleff_error = stats(self.spilleff_mean,self.spilleff_Q,self.spilleff_new,self.iteration)
			self.abseff_mean,self.abseff_Q,self.abseff_error = stats(self.abseff_mean,self.abseff_Q,self.abseff_new,self.iteration)
			self.opteff_mean,self.opteff_Q,self.opteff_error = stats(self.opteff_mean,self.opteff_Q,self.opteff_new,self.iteration)

			self.Square_opteff_mean,self.Square_opteff_Q,self.Square_opteff_error = stats(self.Square_opteff_mean,self.Square_opteff_Q,self.Square_opteff_new,self.iteration)
			self.Square_spilleff_mean,self.Square_spilleff_Q,self.Square_spilleff_error = stats(self.Square_spilleff_mean,self.Square_spilleff_Q,self.Square_spilleff_new,self.iteration)

			self.hist_mean,self.hist_Q,self.hist_error = stats(self.hist_mean,self.hist_Q,self.hist_new,self.iteration)

			#self.peakflux_mean,self.peakflux_Q,self.peakflux_error = stats(self.peakflux_mean,self.peakflux_Q,self.peakflux_new,self.iteration)
			self.peakflux_mean, self.peakflux_error = search_final(self.hist_mean,self.hist_error)

			print("Cos Eff = "+str(self.coseff_mean)+" +/- "+str(3.0*self.coseff_error)+"")   
			print("Shade Eff = "+str(self.shadeeff_mean)+" +/- "+str(3.0*self.shadeeff_error)+"")  
			print("Refl Eff = "+str(self.refleff_mean)+" +/- "+str(3.0*self.refleff_error)+"")  	
			print("Block Eff = "+str(self.blockeff_mean)+" +/- "+str(3.0*self.blockeff_error)+"")  
			print("Spill Eff = "+str(self.spilleff_mean)+" +/- "+str(3.0*self.spilleff_error)+"") 
			print("Absorb Eff = "+str(self.abseff_mean)+" +/- "+str(3.0*self.abseff_error)+"") 
			print("OptEff = "+str(self.opteff_mean)+" +/- "+str(3.0*self.opteff_error)+"") 
			print("Peakflux = "+str(self.peakflux_mean*1.0e-3)+" +/- "+str(3.0*self.peakflux_error*1.0e-3)+"kW/m2") 

			#print("Spill Eff Array = "+str(100.0*self.Square_spilleff_mean)+" %")
			#print("Optical Eff Array = "+str(100.0*self.Square_opteff_mean)+" %")
			print("Processed "+str(self.total_rays)+" rays.")
			print(" ")

			self.error_pct = 3.0*100.0*self.opteff_error/self.opteff_mean
			#self.error_pct = 3.0*100.0*self.peakflux_error/self.peakflux_mean
			print("Max 3 Sigma Error of opteff of 8mx8m square is now "+str(self.error_pct)+" %"+" Need to be < "+str(self.precision))
			self.iteration += 1
		save_hist(self.hist_mean,self.extent,index=str(int(N.degrees(self.azimuth)))+"_"+str(int(90.0-N.degrees(self.elevation))),filename="Fluxmaps",dpi=250)

class SingleCase():
	def __init__(self,recv_area,recv_centre,heliostat,helio_w=1.85,helio_h=2.44,helio_abs=0.10,helio_sigmaxy=1.5e-3,helio_tracking="TiltRoll",bins=100,rays_per_run=100000,precision=1.0,render=False,azimuth=0.0,elevation=45.0):
		self.azimuth = azimuth #input is in degrees
		self.elevation = elevation #input is in degrees
		self.rays_per_run = rays_per_run
		self.precision = precision
		self.render = render
		self.bins = bins

		self.recv_area = recv_area
		self.recv_centre = recv_centre
		self.heliostat = heliostat
		self.helio_w = helio_w
		self.helio_h = helio_h
		self.helio_abs = helio_abs
		self.helio_sigmaxy = helio_sigmaxy
		self.helio_tracking = helio_tracking
		self.bins = bins
		self.rays_per_run = rays_per_run
		self.precision = precision
		self.render = render

	def run(self):
		filename = "PS10_"
		self.stats_file = open(filename+"_SingleRun.csv","w")
		#self.stats_file.write("Azimuth,Elevation,rays,cos,shade,refl,block,spill,absorb,opt,max_flux,cos_se,shade_se,refl_se,block_se,spill_se,abs_se,opt_se,maxflux_se"+"\n")


		stats_string = str(self.azimuth)+","+str(self.elevation)
		S = Stochastic(self.recv_area,self.recv_centre,self.heliostat,N.radians(self.azimuth),N.radians(self.elevation),self.helio_w,self.helio_h,self.helio_abs,self.helio_sigmaxy,self.helio_tracking,self.bins,self.rays_per_run,self.precision,self.render)
		S.trace()
		stats_string += ","+str(S.total_rays)+","+str(S.coseff_mean)+","+str(S.shadeeff_mean)+","+str(S.refleff_mean)+","+str(S.blockeff_mean)+","+str(S.spilleff_mean)+","+str(S.abseff_mean)+","+str(S.opteff_mean)+","+str(S.peakflux_mean)+","+str(S.coseff_error)+","+str(S.shadeeff_error)+","+str(S.refleff_error)+","+str(S.blockeff_error)+","+str(S.spilleff_error)+","+str(S.abseff_error)+","+str(S.opteff_error)+","+str(S.peakflux_error)+"\n"
		self.stats_file.write(stats_string)
		self.stats_file.close()
		




class Hemisphere():
	def __init__(self,recv_area,recv_centre,heliostat,helio_w=1.85,helio_h=2.44,helio_abs=0.10,helio_sigmaxy=1.5e-3,helio_tracking="TiltRoll",bins=100,rays_per_run=100000,precision=1.0,render=False):
		self.rays_per_run = rays_per_run
		self.precision = precision
		self.render = render
		self.bins = bins

		self.recv_area = recv_area
		self.recv_centre = recv_centre
		self.heliostat = heliostat
		self.helio_w = helio_w
		self.helio_h = helio_h
		self.helio_abs = helio_abs
		self.helio_sigmaxy = helio_sigmaxy
		self.helio_tracking = helio_tracking
		self.bins = bins
		self.rays_per_run = rays_per_run
		self.precision = precision
		self.render = render

		self.azi_steps = 18 #36
		self.ele_steps = 9 #8 [-24,24]

	def tabulate(self):
		filename = "PS10_"


		self.azi_interval = 360.0/self.azi_steps
		self.ele_interval = 90.0/self.ele_steps
		
		self.azi_linspace = (N.linspace(0.0,360.0,num=self.azi_steps+1))
		print("Generating 2D table with dimensions (elevation,azimuth)",self.ele_interval,self.azi_interval)

		self.opteff_file = open(filename+"_OptEff.txt",'w')
		self.opteff_file.write("#1"+"\n")
		self.opteff_file.write("float table2D_1("+str(self.ele_steps+2)+","+str(self.azi_steps+2)+")\n")
		self.opteff_file.write(tableline(0.0,self.azi_linspace))

		self.coseff_file = open(filename+"_CosEff.txt",'w')
		self.coseff_file.write("#1"+"\n")
		self.coseff_file.write("float table2D_1("+str(self.ele_steps+2)+","+str(self.azi_steps+2)+")\n")
		self.coseff_file.write(tableline(0.0,self.azi_linspace))

		self.shadeeff_file = open(filename+"_ShadeEff.txt",'w')
		self.shadeeff_file.write("#1"+"\n")
		self.shadeeff_file.write("float table2D_1("+str(self.ele_steps+2)+","+str(self.azi_steps+2)+")\n")
		self.shadeeff_file.write(tableline(0.0,self.azi_linspace))

		self.blockeff_file = open(filename+"_BlockEff.txt",'w')
		self.blockeff_file.write("#1"+"\n")
		self.blockeff_file.write("float table2D_1("+str(self.ele_steps+2)+","+str(self.azi_steps+2)+")\n")
		self.blockeff_file.write(tableline(0.0,self.azi_linspace))
	
		self.spilleff_file = open(filename+"_SpillEff.txt",'w')
		self.spilleff_file.write("#1"+"\n")
		self.spilleff_file.write("float table2D_1("+str(self.ele_steps+2)+","+str(self.azi_steps+2)+")\n")
		self.spilleff_file.write(tableline(0.0,self.azi_linspace))

		self.stats_file = open(filename+"_Stats.csv","w")
		#self.stats_file.write("Azimuth,Elevation,cos,shade,refl,block,spill,absorb,opt,cos_se,shade_se,refl_se,block_se,spill_se,abs_se,opt_se"+"\n")
		#Add the len=10 arrays
		self.stats_file.write("Azimuth,Elevation,cos,shade,refl,block,spill,absorb,opt,cos_se,shade_se,refl_se,block_se,spill_se,abs_se,opt_se,"+stringed(N.arange(1,11))+","+stringed(N.arange(1,11))+"\n") #spill then opt

		#DECLINATION BECOMES ELEVATION AND HRA BECOMES AZIMUTH
		elevation = 0.0
		while elevation <= 90.0:
			#avoid problems where instead of 0, the value of declination is very small
			#if abs(elevation) < 0.01:
				#elevation = 0.0
			#Done
			opteff_line = []
			coseff_line = []
			shadeeff_line = []
			blockeff_line = []
			spilleff_line = []

			if elevation == 90.0: #Then azimuth is irrelevant set to 0.0
				S = Stochastic(self.recv_area,self.recv_centre,self.heliostat,N.radians(0.0),N.radians(elevation),self.helio_w,self.helio_h,self.helio_abs,self.helio_sigmaxy,self.helio_tracking,self.bins,self.rays_per_run,self.precision,self.render)
				S.trace()

				stats_string = str(0.0)+","+str(elevation)
				stats_string += ","+str(S.coseff_mean)+","+str(S.shadeeff_mean)+","+str(S.refleff_mean)+","+str(S.blockeff_mean)+","+str(S.spilleff_mean)+","+str(S.abseff_mean)+","+str(S.opteff_mean)+","+str(S.coseff_error)+","+str(S.shadeeff_error)+","+str(S.refleff_error)+","+str(S.blockeff_error)+","+str(S.spilleff_error)+","+str(S.abseff_error)+","+str(S.opteff_error)+"\n"
				self.stats_file.write(stats_string)

				for azi_point in self.azi_linspace:
					opteff_line.append(S.opteff_mean)
					coseff_line.append(S.coseff_mean)
					shadeeff_line.append(S.shadeeff_mean)
					blockeff_line.append(S.blockeff_mean)
					spilleff_line.append(S.spilleff_mean)

				self.opteff_file.write(tableline(elevation,opteff_line))
				self.coseff_file.write(tableline(elevation,coseff_line))
				self.shadeeff_file.write(tableline(elevation,shadeeff_line))
				self.blockeff_file.write(tableline(elevation,blockeff_line))
				self.spilleff_file.write(tableline(elevation,spilleff_line))

			else:
				for azi_point in self.azi_linspace:
				#calculate azimuth and elevation
				#avoid problems where instead of 0, the value of hra is very small
					if abs(azi_point) < 0.01:
						azi_point = 0.0
				#Done
				#azimuth,elevation = solar_angles(declination,hra_point,self.latitude)
				
					if elevation <= 1.0: #Then the result should just be 0.0 for all
						opteff_line.append(0.0)
						coseff_line.append(0.0)
						shadeeff_line.append(0.0)
						blockeff_line.append(0.0)
						spilleff_line.append(0.0)


				
					else: #Trace Normally
						S = Stochastic(self.recv_area,self.recv_centre,self.heliostat,N.radians(azi_point),N.radians(elevation),self.helio_w,self.helio_h,self.helio_abs,self.helio_sigmaxy,self.helio_tracking,self.bins,self.rays_per_run,self.precision,self.render)
						S.trace()					
						stats_string = str(azi_point)+","+str(elevation)
						stats_string += ","+str(S.coseff_mean)+","+str(S.shadeeff_mean)+","+str(S.refleff_mean)+","+str(S.blockeff_mean)+","+str(S.spilleff_mean)+","+str(S.abseff_mean)+","+str(S.opteff_mean)+","+str(3.0*S.coseff_error)+","+str(3.0*S.shadeeff_error)+","+str(3.0*S.refleff_error)+","+str(3.0*S.blockeff_error)+","+str(3.0*S.spilleff_error)+","+str(3.0*S.abseff_error)+","+str(3.0*S.opteff_error)+","+stringed(S.Square_spilleff_mean)+","+stringed(S.Square_opteff_mean)+"\n"
						self.stats_file.write(stats_string)
						opteff_line.append(S.opteff_mean)
						coseff_line.append(S.coseff_mean)
						shadeeff_line.append(S.shadeeff_mean)
						blockeff_line.append(S.blockeff_mean)
						spilleff_line.append(S.spilleff_mean)

				self.opteff_file.write(tableline(elevation,opteff_line))
				self.coseff_file.write(tableline(elevation,coseff_line))
				self.shadeeff_file.write(tableline(elevation,shadeeff_line))
				self.blockeff_file.write(tableline(elevation,blockeff_line))
				self.spilleff_file.write(tableline(elevation,spilleff_line))


			elevation += self.ele_interval
		self.opteff_file.close()


		self.coseff_file.close()
		self.shadeeff_file.close()
		self.blockeff_file.close()
		self.spilleff_file.close()
		self.stats_file.close()

#Tower Position
tower_x = 0.0
tower_y = 0.0
tower_height = 115.0 #metres

#Heliostat Layout #The csv file is X, Y, Z, focal_length (m)
helio_file = "PS10_Flipped_Slant.csv" #Flipped to suit the southern hemisphere. Mirror focal length is equal to slant range (ideal)
helio_w = 12.925
helio_h = 9.575
tracking = "AzEl"

#Heliostat Properties
helio_abs = 0.05
helio_slope = 1.5e-3 #mrad

#Receiver Properties assumed to be blackbody
recv_width = 8.0
recv_height = 8.0

#Precision
bins = 20 #number of bins on an axis
precision = 3.0 #percentage of 3sigma confidence interval of optical efficiency that is acceptable

#Rendering
render = False #Do we render the heliostat field?

#Tabulation
azi_steps = 36
ele_steps = 9



H = Hemisphere(N.array([recv_width,recv_height]),N.array([tower_x,tower_y,tower_height]),helio_file,helio_w=helio_w,helio_h=helio_h,helio_abs=helio_abs,helio_sigmaxy=helio_slope,helio_tracking=tracking,bins=bins,rays_per_run=100000,precision=precision,render=False)
H.azi_steps = azi_steps
H.ele_steps = ele_steps
H.tabulate()
