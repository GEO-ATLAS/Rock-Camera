import numpy as np
from VirtualCamera.utils import *
from VirtualCamera.Line import Line
class Plane:
    def __init__(self,p0=None,normal=None) -> None:
        # if len(normal)!=3:
        #     raise ValueError("normal size must be 3")
        # if len(p0)!=3:
        #     raise ValueError("p0 size must be 3")
        self.p0=p0
        self.normal=normalize(normal)
        self.ptype="quad"
        self.id=None

    def check(self):
        if self.normal is None:
            raise ValueError("normal has not defined")
        if self.p0 is None:
            raise ValueError("p0 has not defined")
        
        
    def plot(self,ax,width=10,height=10,**kwargs):
        self.check()
        return plotPlane(ax=ax,p0=self.p0,pNormal=self.normal,width=width,height=height,**kwargs)
    
    def intersectPlane(self,plane):
        self.check()
        if not isinstance(plane,Plane):
            raise ValueError("input plane is not instance of Plane")
        plane.check()
        if plane.ptype=="circle":
            flag,p1,p2=planeIntersectCircle(self.p0,self.normal,plane.p0,plane.normal,plane.radius)
            direction=p2-p1
            intersection_point=(p1+p2)/2
        if plane.ptype=="quad":
            flag,intersection_point,direction=planeIntersectPlane(self.p0,self.normal,plane.p0,plane.normal)
        if flag:
            return Line(intersection_point,direction)
        else:
            return None
        
    def intersectCirplane(self,circlePlane):
        self.check()
        if not isinstance(circlePlane,CirclePlane):
            raise ValueError("input circlePlane is not instance of CirclePlane")
        if circlePlane.ptype!="circle":
            raise ValueError("input plane must be circlePlane")
        circlePlane.check()

        return planeIntersectCircle(self.p0,self.normal,circlePlane.p0,circlePlane.normal,circlePlane.radius)
        
class CirclePlane(Plane):
    def __init__(self, p0=None, normal=None, radius=None) -> None:
        super().__init__(p0, normal)
        self.radius=radius
        self.ptype="circle"

    def plot(self,ax,**kwargs):
        return plotCircelPlane(ax,self.p0,self.normal,self.radius,**kwargs)

    
if __name__=="__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim3d(-10,10)
    pl1=Plane([0,0,0],[0,0,1])
    pl2=Plane([0,0,2],[0,1,1])
    pl1.plot(ax,facecolors='r', alpha=0.3)
    pl2.plot(ax,facecolors='b', alpha=0.3)
    cp1=CirclePlane([0,0,1],[1,0,1],10)
    cp1.plot(ax,color="g",alpha=0.3)
    line1=pl1.intersectPlane(pl2)
    line2=pl1.intersectPlane(cp1)
    line3=pl2.intersectPlane(cp1)
    lines=[line1,line2,line3]
    for line in lines:
        if line:
            line.plot(length=10,ax=ax,color="g")
            plotPoint(line.p0,ax,color="r")
    #平面与圆心平面相交的交点
    f1,p11,p12=pl1.intersectCirplane(cp1)
    f2,p21,p22=pl2.intersectCirplane(cp1)
    if f1:
        plotPoint(p11,ax,color="y")
        plotPoint(p12,ax,color="y")
    if f2:
        plotPoint(p21,ax,color="b")
        plotPoint(p22,ax,color="b")
    plt.show()


    