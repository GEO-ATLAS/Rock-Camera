import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from VirtualCamera.Plane import *

class Scene:
    def __init__(self) -> None:
        self.Fractures=[] #fracture plane
        self.rseed=None
        self.regionMinCorner=[0,0,0]
        self.regionMaxCorner=[10,30,30]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=0,azim=-270)
        self.ax=ax
        self.FracturesIntersectTargetPlane=[]

    def updateAxBounding(self,offsetX=10,offsetY=10,offsetZ=10):
        #as cube
        self.ax.set_xlim(self.regionMinCorner[0]-offsetX,self.regionMaxCorner[0]+offsetX)
        self.ax.set_ylim(self.regionMinCorner[1]-offsetY,self.regionMaxCorner[1]+offsetY)
        self.ax.set_zlim3d(self.regionMinCorner[2]-offsetZ,self.regionMaxCorner[2]+offsetZ)

    def updatePlot(self):
        pass

    def setRseed(self,seed):
        self.rseed=seed

    def getRseed(self):
        if self.rseed is None:
            self.rseed=int(datetime.now().timestamp())
        return self.rseed

    @property
    def FractureSize(self):
        return len(self.Fractures)
    
    def addCircularFracture(self,center,dipDirection,dipAngle,radius):
        '''定义中心点，倾向、倾角、半径，添加圆形裂隙'''
        if not (isinstance(center,list) or isinstance(center,np.ndarray)):
            raise ValueError("center wrong!")
        dipDirection=np.deg2rad(dipDirection)
        dipAngle=np.deg2rad(dipAngle)
        normal,kapa=self.__dipDirectionAngleToNormal__(dipDirection,dipAngle)
        normal=normalize(normal)
        c=CirclePlane(center,normal,radius)
        c.id=self.FractureSize+1
        self.Fractures.append(c)
        return c

    def addCircularFractureByUnitVec(self,center,normal,radius):
        c=CirclePlane(center,normal,radius)
        c.id=self.FractureSize+1
        self.Fractures.append(c)
        return c

    def __dipDirectionAngleToNormal__(self,dipDirection,dipAngle):
        '''倾向倾角转法向量'''
        kapa = 0
        echis = (np.pi / 2) - dipAngle
        if 0 <= dipDirection < np.pi:
            kapa = dipDirection + np.pi
        if np.pi <= dipDirection <= 2 * np.pi:
            kapa = dipDirection - np.pi    
        poleMean = np.array([np.cos(kapa) * np.cos(echis),
                            -np.sin(kapa) * np.cos(echis),
                            -np.sin(echis)])
        return poleMean,kapa
    
    def addBaecherFracture(self,meanDipDirection,meanDipAngle,fisherConstant,sizeDistribution,meanFractureSize,sigmaFractureSize):
        '''
            input degree
            fisherConstant 需介于>0 or 2<
            sizeDistribution:尺寸大小的分布模型，log、exp、det
        '''
        assert sizeDistribution in ["log", "exp", "det"]
        assert fisherConstant >= 2 or fisherConstant <= 0
        meanDipDirection=np.deg2rad(meanDipDirection)
        meanDipAngle=np.deg2rad(meanDipAngle)
        np.random.seed(self.getRseed())
        self.rseed+=1
        poleAllRotated = np.zeros(3)
        if fisherConstant >= 2:                
            poleMean,kapa=self.__dipDirectionAngleToNormal__(meanDipDirection,meanDipAngle)
            assert np.allclose(np.linalg.norm(poleMean), 1)
            
            fisherDipDevAngle = np.arccos((fisherConstant + np.log(1 - np.random.rand())) / fisherConstant)
            echis = (np.pi / 2) - (meanDipAngle - fisherDipDevAngle)
            poleDipRotated = np.array([np.cos(kapa) * np.cos(echis),
                                    -np.sin(kapa) * np.cos(echis),
                                    -np.sin(echis)])
            randomAngle = randomize(2 * np.pi)# random
            
            rotCorrectionUnitVec = rotateOnAxis2(randomAngle,poleMean)
            rotCorrectionUnitVec =normalize(rotCorrectionUnitVec)
            poleAllRotated = rotCorrectionUnitVec.dot(poleDipRotated)
            poleAllRotated=normalize(poleAllRotated)
            assert np.allclose(np.linalg.norm(poleAllRotated), 1)

        elif fisherConstant <= 0:
            randomDipAngle = randomize(np.pi/2)
            randomDipDirection = randomize(2 * np.pi)
            poleAllRotated,kapa=self.__dipDirectionAngleToNormal__(randomDipDirection,randomDipAngle)
            assert np.allclose(np.linalg.norm(poleAllRotated), 1)
            
        # random fracture location (poisson process)
        randCoordinateX = self.regionMinCorner[0] + np.random.rand() * (self.regionMaxCorner[0] - self.regionMinCorner[0])
        randCoordinateY = self.regionMinCorner[1] + np.random.rand() * (self.regionMaxCorner[1] - self.regionMinCorner[1])
        randCoordinateZ = self.regionMinCorner[2] + np.random.rand() * (self.regionMaxCorner[2] - self.regionMinCorner[2])
        assert randCoordinateX > 0 and randCoordinateY > 0 and randCoordinateZ > 0
        assert not np.isclose(randCoordinateX, 0) and not np.isclose(randCoordinateY, 0) and not np.isclose(randCoordinateZ, 0)
        newFracCenter = np.array([randCoordinateX, randCoordinateY, randCoordinateZ])
        
        # add circular fracture
        if sizeDistribution == "log":
            return self.addCircularFractureByUnitVec(newFracCenter, poleAllRotated,np.random.lognormal(meanFractureSize,sigmaFractureSize))
        if sizeDistribution == "exp":
            return self.addCircularFractureByUnitVec(newFracCenter, poleAllRotated, np.random.exponential(1/meanFractureSize))
        if sizeDistribution == "det":
            return self.addCircularFractureByUnitVec(newFracCenter, poleAllRotated, meanFractureSize)


    def plot(self):
        for fracture in self.Fractures:
            fracture.plot(self.ax,color="b")

    def setTargetPlane(self,plane:Plane):
        self.TargetPlane=plane
    
    def calFractureIntersectTargetPlane(self):
        '''cal fracture intersect TargetPlane'''
        self.FracturesIntersectTargetPlane=[]
        if self.TargetPlane is None or (not isinstance(self.TargetPlane,Plane)):
            raise ValueError("targetPlane hadn't been defined!")
        for frac in self.Fractures:
            flag,p1,p2=self.TargetPlane.intersectCirplane(frac)
            if flag:
                self.FracturesIntersectTargetPlane.append([frac,p1,p2])
        return self.FracturesIntersectTargetPlane
    
    def plotIntersectTargetPlane(self):
        if len(self.FracturesIntersectTargetPlane)==0:
            self.calFractureIntersectTargetPlane()
        for frac in self.Fractures:
            frac.plot(self.ax)
        for res in self.FracturesIntersectTargetPlane:
            [frac,p1,p2]=res
            #frac.plot(self.ax)
            plotPoint(p1,self.ax)
            plotPoint(p2,self.ax)
            plotLine(p1,p2,self.ax)

    def plotRegionBounding(self):
        face_normals,face_centers,face_widths,face_heights=calRegionFace(self.regionMinCorner,self.regionMaxCorner)
        for i in range(6):
            n=face_normals[i]
            c=face_centers[i]
            w=face_widths[i]
            h=face_heights[i]
            p=Plane(c,n)
            p.plot(self.ax,w,h,alpha=0.3)

if __name__=="__main__":
    scene=Scene()
    scene.regionMaxCorner=[30,80,30]
    scene.setRseed(100)
    for i in range(12):
        plane=Plane([15,5+i*5,15],[0,1,0])
        plane.plot(scene.ax,35,35,color="g",alpha=0.2)

    for i in range(50):
        frac=scene.addBaecherFracture(360,90,23,"log",0.725, 0.52)
        # flag,p1,p2=plane.intersectCirplane(frac)
        # if flag:
        #     plotPoint(p1,scene.ax)
        #     plotPoint(p2,scene.ax)
        #     plotLine(p1,p2,scene.ax,color="g")
        #     frac.plot(scene.ax,color="b",alpha=0.3)

    
    scene.setTargetPlane(plane)
    #
    scene.plotIntersectTargetPlane()
    scene.plotRegionBounding()

    scene.ax.set_xlim(0,30)
    scene.ax.set_ylim(0,50)
    scene.ax.set_zlim3d(0,30)
    scene.ax.set_axis_off()
    #scene.plot()
    plt.show()





