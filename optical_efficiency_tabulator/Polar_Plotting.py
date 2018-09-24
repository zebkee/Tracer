# Plots the optical efficiency files in this directory using polar coordinate system.
# Also overlays sunpath for a given latitude.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv

def ThetaRValues(filename,index):
	"""Generates values of theta, r and efficiency by reading a file"""
	f = open(filename+"_"+index+".txt","r")
	data = f.readlines()
	data = data[2:]
	matrix = []
	i = 2
	for line in data:
		matrix.append(line[0:-2].split(" "))

	#print(matrix)
	matrix = np.array(matrix).astype(np.float)
	print(np.shape(matrix))
	#print(matrix)

	#extract azimuths

	azimuths = (matrix)[0][1:]
	print(azimuths)

	#extract elevations
	elevations = (matrix[:,0])[1:]

	values = (matrix[1:,1:]).T #transpose so it is rows of zeniths stacked up


	zeniths = 90.0 - elevations
	azimuths = np.radians(azimuths)
	r , theta = np.meshgrid(zeniths,azimuths)

	return [theta,r,values]
#Read the file
def Polar(filename,field_info,location,latitude):

	lat = np.radians(latitude)
	HRA_Summer = np.linspace(-1.0*np.arccos(-1.0*np.tan(lat)*np.tan(0.40928)),+1.0*np.arccos(-1.0*np.tan(lat)*np.tan(0.40928)),100)
	HRA_Spring = np.linspace(-1.0*np.arccos(-1.0*np.tan(lat)*np.tan(0.0)),+1.0*np.arccos(-1.0*np.tan(lat)*np.tan(0.0)),100)
	HRA_Winter = np.linspace(-1.0*np.arccos(-1.0*np.tan(lat)*np.tan(-0.40928)),+1.0*np.arccos(-1.0*np.tan(lat)*np.tan(-0.40928)),100)

	Elev_Summer = np.arcsin(np.sin(0.40928)*np.sin(lat)+np.cos(0.40928)*np.cos(lat)*np.cos(HRA_Summer))
	Elev_Spring = np.arcsin(np.sin(0.0)*np.sin(lat)+np.cos(0.0)*np.cos(lat)*np.cos(HRA_Spring))
	Elev_Winter = np.arcsin(np.sin(-0.40928)*np.sin(lat)+np.cos(-0.40928)*np.cos(lat)*np.cos(HRA_Winter))

	Azi_Summer = np.arccos((np.sin(0.40928)*np.cos(lat)-np.cos(0.40928)*np.sin(lat)*np.cos(HRA_Summer))/(np.cos(Elev_Summer)))
	Azi_Spring = np.arccos((np.sin(0.0)*np.cos(lat)-np.cos(0.0)*np.sin(lat)*np.cos(HRA_Spring))/(np.cos(Elev_Spring)))
	Azi_Winter = np.arccos((np.sin(-0.40928)*np.cos(lat)-np.cos(-0.40928)*np.sin(lat)*np.cos(HRA_Winter))/(np.cos(Elev_Winter)))

	i = 0
	while i < len(HRA_Summer):
		if HRA_Summer[i] > 0.0:
			Azi_Summer[i] = 2.0*np.pi-Azi_Summer[i]
		i += 1
	i = 0
	while i < len(HRA_Spring):
		if HRA_Spring[i] > 0.0:
			Azi_Spring[i] = 2.0*np.pi-Azi_Spring[i]
		i += 1
	i = 0
	while i < len(HRA_Winter):
		if HRA_Winter[i] > 0.0:
			Azi_Winter[i] = 2.0*np.pi-Azi_Winter[i]
		i += 1

	Zenith_Summer = 90.0 - np.degrees(Elev_Summer)
	Zenith_Spring = 90.0 - np.degrees(Elev_Spring)
	Zenith_Winter = 90.0 - np.degrees(Elev_Winter)
	#Indices are 1.Cos, 2.Shade, 3.Block, 4.Spill, 5.OptEff
	#7.DNI 8.OptEff
	cos_list = ThetaRValues(filename,"CosEff")
	shade_list = ThetaRValues(filename,"ShadeEff")
	block_list = ThetaRValues(filename,"BlockEff")
	spill_list = ThetaRValues(filename,"SpillEff")
	opt_list = ThetaRValues(filename,"OptEff")

	fig = plt.figure()

	grid = plt.GridSpec(20,34,wspace=0.1,hspace=0.1)

	ax1 = fig.add_subplot(grid[0:4,0:7],projection="polar")
	ax2 = fig.add_subplot(grid[0:4,8:15],projection="polar")
	ax3 = fig.add_subplot(grid[5:9,0:7],projection="polar")
	ax4 = fig.add_subplot(grid[5:9,8:15],projection="polar")
	ax5 = fig.add_subplot(grid[0:9,17:32],projection="polar")
	ax6 = fig.add_subplot(grid[0:9,33]) #1st Colorbar
	ax7 = fig.add_subplot(grid[10:19,0:15],projection = "polar")
	ax8 = fig.add_subplot(grid[10:19,17:32],projection="polar")
	ax9 = fig.add_subplot(grid[10:19,33]) #2nd ColorBar

	ax1.contourf(cos_list[0],cos_list[1],cos_list[2],np.linspace(0.0,1.00,41),extend='neither',cmap="jet")
	ax2.contourf(shade_list[0],shade_list[1],shade_list[2],np.linspace(0.0,1.00,41),extend='max',cmap="jet")
	ax3.contourf(block_list[0],block_list[1],block_list[2],np.linspace(0.0,1.00,41),extend='neither',cmap="jet")
	ax4.contourf(spill_list[0],spill_list[1],spill_list[2],np.linspace(0.0,1.00,41),extend='neither',cmap="jet")
	ax5.contourf(opt_list[0],opt_list[1],opt_list[2],np.linspace(0.0,1.00,41),extend='neither',cmap="jet")

	#Polt interpolation dots
	ax5.plot(opt_list[0],opt_list[1],"ko",markersize=1)

	#Plot Sunpath
	ax1.plot(Azi_Summer,Zenith_Summer,"k-",linewidth=1)
	ax1.plot(Azi_Spring,Zenith_Spring,"k--",linewidth=1)
	ax1.plot(Azi_Winter,Zenith_Winter,"k-",linewidth=1)

	ax2.plot(Azi_Summer,Zenith_Summer,"k-",linewidth=1)
	ax2.plot(Azi_Spring,Zenith_Spring,"k--",linewidth=1)
	ax2.plot(Azi_Winter,Zenith_Winter,"k-",linewidth=1)

	ax3.plot(Azi_Summer,Zenith_Summer,"k-",linewidth=1)
	ax3.plot(Azi_Spring,Zenith_Spring,"k--",linewidth=1)
	ax3.plot(Azi_Winter,Zenith_Winter,"k-",linewidth=1)

	ax4.plot(Azi_Summer,Zenith_Summer,"k-",linewidth=1)
	ax4.plot(Azi_Spring,Zenith_Spring,"k--",linewidth=1)
	ax4.plot(Azi_Winter,Zenith_Winter,"k-",linewidth=1)

	ax5.plot(Azi_Summer,Zenith_Summer,"k-",linewidth=1)
	ax5.plot(Azi_Spring,Zenith_Spring,"k--",linewidth=1)
	ax5.plot(Azi_Winter,Zenith_Winter,"k-",linewidth=1)

	#DNI Calculaions
	#Remember that zeniths are in degrees
	AM = (np.cos(np.radians(opt_list[1])))**-1.0
	DNI = 1367*(0.7**(AM**0.678))

	ax7.contourf(opt_list[0],opt_list[1],DNI,np.linspace(0.0,1000.0,41),extend='neither',cmap="jet")
	ax7.plot(Azi_Summer,Zenith_Summer,"k-",linewidth=1)
	ax7.plot(Azi_Spring,Zenith_Spring,"k--",linewidth=1)
	ax7.plot(Azi_Winter,Zenith_Winter,"k-",linewidth=1)

	ax8.contourf(opt_list[0],opt_list[1],opt_list[2]*DNI,np.linspace(0.0,1000.0,41),extend='neither',cmap="jet")
	ax8.plot(Azi_Summer,Zenith_Summer,"k-",linewidth=1)
	ax8.plot(Azi_Spring,Zenith_Spring,"k--",linewidth=1)
	ax8.plot(Azi_Winter,Zenith_Winter,"k-",linewidth=1)
	
	#Labels and hide the zeniths
	ax1.set_yticklabels([])
	ax2.set_yticklabels([])
	ax3.set_yticklabels([])
	ax4.set_yticklabels([])
	ax1.set_xticklabels([])
	ax2.set_xticklabels([])
	ax3.set_xticklabels([])
	ax4.set_xticklabels([])

	ax1.set_title("Cos")
	ax2.set_title("Shade")
	ax3.set_title("Block")
	ax4.set_title("Spill")
	ax5.set_title("OptEff")
	ax7.set_title(r"DNI = $1367 \times 0.7^{AM^{0.678}}$")
	ax8.set_title("DNI*OptEff")
	

	fig.suptitle("Collector_RecvArea = "+field_info+"   "+"Location = "+location+"\n"+"Latitude = "+str(np.degrees(lat)),fontsize="x-large")

	cNorm1 = matplotlib.colors.Normalize(vmin=0.0,vmax=1.0)
	cNorm2 = matplotlib.colors.Normalize(vmin=0.0,vmax=1000.0)
	cbar1 = matplotlib.colorbar.ColorbarBase(ax6, norm=cNorm1,cmap="jet",ticks = np.linspace(0.0,1.0,21),label="Efficiency")
	cbar2 = matplotlib.colorbar.ColorbarBase(ax9, norm=cNorm2,cmap="jet",ticks = np.linspace(0.0,1000.0,21),label="Power (W/m2)")

	fig.set_size_inches(8.27,11.69)
	plt.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.10)
	plt.savefig("PolarContour.png",dpi=100)
	return

filename = "PS10_"
field_info = "PS10_AzEl_8mx8m"
location = "Alice Springs (AUS)"
latitude = -23.795


Polar(filename,field_info,location,latitude)
