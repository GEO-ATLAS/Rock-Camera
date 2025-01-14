#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       虚拟相机变角度拍摄测试
@Date     :2023/10/10 17:07:36
@Author      :chenbaolin
@version      :1.0
'''
from UBG.unblocks import *
import os
from VirtualCamera.CalibCamera import Camera
from VirtualCamera.Plane import *
import matplotlib.pyplot as plt 
modelSaveDir="./testCanon2/model"
imageSaveDir="./testCanon2/"

if not os.path.exists(modelSaveDir):
	os.makedirs(modelSaveDir)
#准备相机
camera=Camera(position=[7.5,7.5,5],LookAt=[7.5,7.5,1000])#初始化
camera.frustumDepth=2
camera.focalLength=20 #调整焦距
camera.ResX=6000 #pixel 图像的成像尺寸 X
camera.ResY=4000 #pixel 
camera.SensorWidth=22.3 #mm 传感器参数
camera.SensorHeight=14.9 #mm

camera.getFov()
#创建DFN空间
dfn = DFN()
dfn.set_RandomSeed(100)#设置随机数-则相同随机数下生成的模型是相同的
dfn.set_RegionMaxCorner([15,15,150])
dfn.add_FractureSet()#添加裂隙集
fracCenter=[7.5,7.5,10]#裂隙的中心位置
fracdipDirection=90
fracdipAngle=90
fracRadius=5
#添加两个互相垂直的裂隙面
dfn.fractureSets[0].add_CircularFracture(fracCenter,180,fracdipAngle,fracRadius)
dfn.fractureSets[0].add_CircularFracture(fracCenter,90,fracdipAngle,fracRadius)
generator = Generator()#块体生成器

#按焦距创建目录
imageSaveDir=imageSaveDir+str(camera.focalLength)
if not os.path.exists(imageSaveDir):
	os.makedirs(imageSaveDir)
stepSize=0.5 #步长
observeDist=11 #相机与目标平面距离保持固定
start=5.5 #开始位置，这个不影响拍摄,为了确保目标平面与裂隙面相交，fracCenterZ-fracRadius
cameraHeight=7.5 #初始高度相机高度
observeAngle=0#相机拍摄角度，仅竖直方向Y,此处保持水平

	
for s in range(21):
	targetZ=stepSize*s+start
	imageSavePath=imageSaveDir+f"/{targetZ}/"
	os.makedirs(imageSavePath,exist_ok=True)
	for angle in range(-45,45,5):
		#相机与目标平面距离保持固定
		camera.position=[7.5,cameraHeight,targetZ-observeDist]
		camera.LookAt=[7.5,cameraHeight+observeDist*np.tan(np.deg2rad(angle)),targetZ]
		#目标平面
		tpCenter=[7.5,7.5,targetZ]
		tp1=[15,15,targetZ]
		tp2=[15,0,targetZ]
		tp3=[0,0,targetZ]
		tp4=[0,15,targetZ]
		tpNormal=[0,0,1]
		tPlane=Plane(tpCenter,tpNormal)
		#遍历每个生成的裂隙集及其裂隙面
		lines=[]
		
		for i in range(dfn.FractureSetsNum):
			fs=dfn.fractureSets[i]
			n=fs.get_FractureNum() #集合中的裂隙数量
			for k in range(n):
				f=fs.fractures[k]#获取裂隙面
				#计算裂隙面与目标平面的交点
				#print("id",f.id,"Center:",f.get_Center(),"normal:",f.get_UnitVector(),"radius:",f.get_BoundingSphereRadius(),"area:",f.get_Area())
				frac=CirclePlane(f.get_Center(),f.get_UnitVector(),f.get_BoundingSphereRadius())
				flag,p1,p2=tPlane.intersectCirplane(frac)
				if flag>0:
					lines.append([p1,p2])
		#目标平面位置的dfn空间横切面
		lines.append([tp1,tp2])
		lines.append([tp2,tp3])
		lines.append([tp3,tp4])
		lines.append([tp4,tp1])
		print("camera_position",s,camera.position,tPlane.p0)
		
		camera.CaptureLines(lines,savePath=imageSavePath+"/"+str(round(angle,2))+"_.jpg" ,thickness=10)
	
	targetPlaneFileName="plane_"+str(round(targetZ,2))
	generator.generate_PlaneVTK(modelSaveDir+"/"+targetPlaneFileName,tpCenter,tpNormal,45,45)#生成平面

dfnModelFileName="dfn"
dfn.export_DFNVtk(modelSaveDir+"/"+dfnModelFileName)#保存相应的模型
dfn.export_RegionVtk(modelSaveDir+"/region")

generator.set_MinInscribedSphereRadius(0.05)#参数设置 
generator.set_MaxAspectRatio(30)
generator.generate_RockMass(dfn)#执行生成
generator.export_BlocksVtk(modelSaveDir+"/block")

plt.show()