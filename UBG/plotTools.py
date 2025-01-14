import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import imread
import math 
import os
def showPlots(): 
	plt.show()

def blockVolumeDistribution(x):
	plt.figure(1)
	x.sort()
	y = []
	totalVolume = 0;
	acumVol = []
	for i in range(len(x)): 
		totalVolume += x[i]
		if (i>0): acumVol.append(acumVol[i-1] + x[i])
		if (i==0): acumVol.append(x[i])
	for i in range(len(x)):
		y.append(100.0*acumVol[i]/totalVolume) 
	plt.xlabel('Block Volume (m3)', fontweight='bold')
	plt.ylabel('% Volume Smaller Than', fontweight='bold')
	plt.title('Block Volume Distribution', fontweight='bold')
	plt.grid(True)
	plt.gca().xaxis.grid(True)
	plt.xscale('log')
	plt.xlim(x[0],x[len(x)-1])
	plt.plot(x,y);
	
def blockShapeDiagram(x,y,z,perc=1.0):
	plt.figure(2)
	D20 = 0
	D40 = 0
	D60 = 0
	D80 = 0
	vols = list(z)
	vols.sort()
	volsDistrib = []
	totalVolume = 0;
	acumVol = []
	for i in range(len(vols)): 
		totalVolume += vols[i]
		if (i>0): acumVol.append(acumVol[i-1] + vols[i])
		if (i==0): acumVol.append(vols[i])
	for i in range(len(vols)):
		volsDistrib.append(100.0*acumVol[i]/totalVolume) 
	for i in range (len(volsDistrib)):
		if (volsDistrib[i] <= 20): D20 = vols[i]
		if (volsDistrib[i] <= 40): D40 = vols[i]
		if (volsDistrib[i] <= 60): D60 = vols[i]
		if (volsDistrib[i] <= 80): D80 = vols[i]
	D20Alpha = []
	D20Beta = []
	D20to40Alpha = []
	D20to40Beta = []
	D40to60Alpha = []
	D40to60Beta = []
	D60to80Alpha = []
	D60to80Beta = []
	D80Alpha = []
	D80Beta = []
	for i in range(len(z)):
		if (i % int(1.0/perc) == 0):
			if (z[i] < D20):
				D20Alpha.append(x[i])
				D20Beta.append(y[i])
			if (z[i] >= D20 and z[i] < D40):
				D20to40Alpha.append(x[i])
				D20to40Beta.append(y[i])
			if (z[i] >= D40 and z[i] < D60):
				D40to60Alpha.append(x[i])
				D40to60Beta.append(y[i])
			if (z[i] >= D60 and z[i] < D80):
				D60to80Alpha.append(x[i])
				D60to80Beta.append(y[i])
			if (z[i] >= D80):
				D80Alpha.append(x[i])
				D80Beta.append(y[i])
	for i in range(len(D20Alpha)):
		D20Alpha[i] = math.log10(D20Alpha[i]) * 9 + 1
		D20Alpha[i] = (1 - (1.0/9.0)*(D20Beta[i]-1))*D20Alpha[i] + (5.5/9.0)*(D20Beta[i]-1)
	for i in range(len(D20to40Alpha)):
		D20to40Alpha[i] = math.log10(D20to40Alpha[i]) * 9 + 1
		D20to40Alpha[i] = (1 - (1.0/9.0)*(D20to40Beta[i]-1))*D20to40Alpha[i] + (5.5/9.0)*(D20to40Beta[i]-1)
	for i in range(len(D40to60Alpha)):
		D40to60Alpha[i] = math.log10(D40to60Alpha[i]) * 9 + 1
		D40to60Alpha[i] = (1 - (1.0/9.0)*(D40to60Beta[i]-1))*D40to60Alpha[i] + (5.5/9.0)*(D40to60Beta[i]-1)
	for i in range(len(D60to80Alpha)):
		D60to80Alpha[i] = math.log10(D60to80Alpha[i]) * 9 + 1
		D60to80Alpha[i] = (1 - (1.0/9.0)*(D60to80Beta[i]-1))*D60to80Alpha[i] + (5.5/9.0)*(D60to80Beta[i]-1)
	for i in range(len(D80Alpha)):
		D80Alpha[i] = math.log10(D80Alpha[i]) * 9 + 1
		D80Alpha[i] = (1 - (1.0/9.0)*(D80Beta[i]-1))*D80Alpha[i] + (5.5/9.0)*(D80Beta[i]-1)
	plt.scatter(D20Alpha,D20Beta, color='b', s=2, edgecolor='k', label='<D20')
	plt.scatter(D20to40Alpha,D20to40Beta, color='c', s=7, edgecolor='k', label='D20 to D40')
	plt.scatter(D40to60Alpha,D40to60Beta, color='y', s=13, edgecolor='k', label='D40 to D60')
	plt.scatter(D60to80Alpha,D60to80Beta, color='m', s=18, edgecolor='k', label='D60 to D80')
	plt.scatter(D80Alpha,D80Beta, color='r', s=23, edgecolor='k', label='>D80')
	plt.legend(prop={'size': 12})
	img = imread(os.path.join(os.path.dirname(__file__),"TriangularPlot.png"))
	plt.imshow(img, zorder=0, extent=[0, 11, 0, 11])
	plt.axis('off')

def BlockShapeDistribution(x,y,z):
	plt.figure(3)
	D20 = 0
	D40 = 0
	D60 = 0
	D80 = 0
	vols = list(z)
	vols.sort()
	volsDistrib = []
	totalVolume = 0
	acumVol = []
	for i in range(len(vols)): 
		totalVolume += vols[i]
		if (i>0): acumVol.append(acumVol[i-1] + vols[i])
		if (i==0): acumVol.append(vols[i])
	for i in range(len(vols)):
		volsDistrib.append(100.0*acumVol[i]/totalVolume) 
	for i in range (len(volsDistrib)):
		if (volsDistrib[i] <= 20): D20 = vols[i]
		if (volsDistrib[i] <= 40): D40 = vols[i]
		if (volsDistrib[i] <= 60): D60 = vols[i]
		if (volsDistrib[i] <= 80): D80 = vols[i]
	D20Alpha = []
	D20Beta = []
	D20Volume = []
	D20to40Alpha = []
	D20to40Beta = []
	D20to40Volume = []
	D40to60Alpha = []
	D40to60Beta = []
	D40to60Volume = []
	D60to80Alpha = []
	D60to80Beta = []
	D60to80Volume = []
	D80Alpha = []
	D80Beta = []
	D80Volume = []
	for i in range(len(x)):
		if (z[i] < D20):
			D20Alpha.append(x[i])
			D20Beta.append(y[i])
			D20Volume.append(z[i])
		if (z[i] >= D20 and z[i] < D40):
			D20to40Alpha.append(x[i])
			D20to40Beta.append(y[i])
			D20to40Volume.append(z[i])
		if (z[i] >= D40 and z[i] < D60):
			D40to60Alpha.append(x[i])
			D40to60Beta.append(y[i])
			D40to60Volume.append(z[i])
		if (z[i] >= D60 and z[i] < D80):
			D60to80Alpha.append(x[i])
			D60to80Beta.append(y[i])
			D60to80Volume.append(z[i])
		if (z[i] >= D80):
			D80Alpha.append(x[i])
			D80Beta.append(y[i])
			D80Volume.append(z[i])
	Cbar = []
	CEbar = []
	Ebar = []
	EPbar = []
	Pbar = []
	PCbar = []
	volC = 0
	volCE = 0
	volE = 0
	volEP = 0
	volP = 0
	volPC = 0
	for i in range(len(D20Alpha)):
		alpha = D20Alpha[i]
		beta = D20Beta[i]
		vol = D20Volume[i]
		if (alpha < 2 and beta < 4): volC += vol
		if (alpha < 3 and beta >= 4 and beta < 7): volCE += vol
		if (beta >= 7): volE += vol
		if (alpha >= 3 and beta < 7 and beta >= 4): volEP += vol
		if (alpha >= 5 and beta < 4): volP += vol
		if (alpha >= 2 and alpha < 5 and beta < 4): volPC += vol
	Cbar.append(100.0*volC/totalVolume)
	CEbar.append(100.0*volCE/totalVolume)	
	Ebar.append(100.0*volE/totalVolume)
	EPbar.append(100.0*volEP/totalVolume)
	Pbar.append(100.0*volP/totalVolume)
	PCbar.append(100.0*volPC/totalVolume)
	volC = 0
	volCE = 0
	volE = 0
	volEP = 0
	volP = 0
	volPC = 0
	for i in range(len(D20to40Alpha)):
		alpha = D20to40Alpha[i]
		beta = D20to40Beta[i]
		vol = D20to40Volume[i]
		if (alpha < 2 and beta < 4): volC += vol
		if (alpha < 3 and beta >= 4 and beta < 7): volCE += vol
		if (beta >= 7): volE += vol
		if (alpha >= 3 and beta < 7 and beta >= 4): volEP += vol
		if (alpha >= 5 and beta < 4): volP += vol
		if (alpha >= 2 and alpha < 5 and beta < 4): volPC += vol
	Cbar.append(100.0*volC/totalVolume)
	CEbar.append(100.0*volCE/totalVolume)	
	Ebar.append(100.0*volE/totalVolume)
	EPbar.append(100.0*volEP/totalVolume)
	Pbar.append(100.0*volP/totalVolume)
	PCbar.append(100.0*volPC/totalVolume)
	volC = 0
	volCE = 0
	volE = 0
	volEP = 0
	volP = 0
	volPC = 0
	for i in range(len(D40to60Alpha)):
		alpha = D40to60Alpha[i]
		beta = D40to60Beta[i]
		vol = D40to60Volume[i]
		if (alpha < 2 and beta < 4): volC += vol
		if (alpha < 3 and beta >= 4 and beta < 7): volCE += vol
		if (beta >= 7): volE += vol
		if (alpha >= 3 and beta < 7 and beta >= 4): volEP += vol
		if (alpha >= 5 and beta < 4): volP += vol
		if (alpha >= 2 and alpha < 5 and beta < 4): volPC += vol
	Cbar.append(100.0*volC/totalVolume)
	CEbar.append(100.0*volCE/totalVolume)	
	Ebar.append(100.0*volE/totalVolume)
	EPbar.append(100.0*volEP/totalVolume)
	Pbar.append(100.0*volP/totalVolume)
	PCbar.append(100.0*volPC/totalVolume)
	volC = 0
	volCE = 0
	volE = 0
	volEP = 0
	volP = 0
	volPC = 0
	for i in range(len(D60to80Alpha)):
		alpha = D60to80Alpha[i]
		beta = D60to80Beta[i]
		vol = D60to80Volume[i]
		if (alpha < 2 and beta < 4): volC += vol
		if (alpha < 3 and beta >= 4 and beta < 7): volCE += vol
		if (beta >= 7): volE += vol
		if (alpha >= 3 and beta < 7 and beta >= 4): volEP += vol
		if (alpha >= 5 and beta < 4): volP += vol
		if (alpha >= 2 and alpha < 5 and beta < 4): volPC += vol
	Cbar.append(100.0*volC/totalVolume)
	CEbar.append(100.0*volCE/totalVolume)	
	Ebar.append(100.0*volE/totalVolume)
	EPbar.append(100.0*volEP/totalVolume)
	Pbar.append(100.0*volP/totalVolume)
	PCbar.append(100.0*volPC/totalVolume)
	volC = 0
	volCE = 0
	volE = 0
	volEP = 0
	volP = 0
	volPC = 0
	for i in range(len(D80Alpha)):
		alpha = D80Alpha[i]
		beta = D80Beta[i]
		vol = D80Volume[i]
		if (alpha < 2 and beta < 4): volC += vol
		if (alpha < 3 and beta >= 4 and beta < 7): volCE += vol
		if (beta >= 7): volE += vol
		if (alpha >= 3 and beta < 7 and beta >= 4): volEP += vol
		if (alpha >= 5 and beta < 4): volP += vol
		if (alpha >= 2 and alpha < 5 and beta < 4): volPC += vol
	Cbar.append(100.0*volC/totalVolume)
	CEbar.append(100.0*volCE/totalVolume)	
	Ebar.append(100.0*volE/totalVolume)
	EPbar.append(100.0*volEP/totalVolume)
	Pbar.append(100.0*volP/totalVolume)
	PCbar.append(100.0*volPC/totalVolume)
	labels = ['< D20', 'D20 to D40', 'D40 to D60', 'D60 to D80', '>D80']
	x = np.arange(len(labels)) 
	width = 0.1 
	rects1 = plt.bar(x - 2.5*width, Cbar, width, edgecolor='k', label='C - Cubic')
	rects2 = plt.bar(x - 1.5*width, CEbar, width, edgecolor='k', label='CE - Transitional Shape')
	rects3 = plt.bar(x - 0.5*width, Ebar, width, edgecolor='k', label='E - Elongated')
	rects4 = plt.bar(x + 0.5*width, EPbar, width, edgecolor='k',label='EP - Transitional Shape')
	rects5 = plt.bar(x + 1.5*width, Pbar, width, edgecolor='k', label='P - Platy')
	rects6 = plt.bar(x + 2.5*width, PCbar, width, edgecolor='k', label='PC - Transitional Shape')
	plt.xlabel('Block Volume Bin', fontweight='bold')
	plt.ylabel('% of Total Volume', fontweight='bold')
	plt.title('Shape Distribution', fontweight='bold')
	plt.xticks(x, labels, fontweight='bold')
	plt.legend()
	plt.grid(True)
	plt.gca().set_axisbelow(True) 
	plt.gca().xaxis.grid(False)
	plt.gca().yaxis.grid(True)
