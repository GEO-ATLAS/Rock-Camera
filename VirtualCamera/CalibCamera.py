import numpy as np
import matplotlib.pyplot as plt
from VirtualCamera.utils import *
import os
from VirtualCamera.Line import Line 
DEFAULT_LOOKDIRECTION=[0,0,1]
class Camera(object):
    def __init__(self,position,LookAt=[0,10,0],img_ax=None) -> None:
        self.K=np.eye(4)
        self.Rot=np.eye(4)
        self.RT=np.eye(4)
        self.position=position #m
        self.LookAt=LookAt
        self.focalLength=35 #mm
        self.ResX=4256 #pixel
        self.ResY=2832 #pixel
        self.SensorWidth=36.0 #mm
        self.SensorHeight=23.6 #mm
        self.s=0
        self.cameraWidth=None #模型
        self.cameraHeight=None
        self.frustumDepth=100
        self.projectImageDrawBound=False
        
        if img_ax is None:
            fig2 = plt.figure(1)
            img_ax = fig2.add_subplot(111)
        self.img_ax=img_ax

    def setLookAt(self,LookAt):
        self.LookAt=LookAt
        self.getRT()

    def getLookAt(self):
        return self.LookAt
    
    def setPosition(self,position):
        self.position=position
        self.getRT()
    def getPosition(self):
        return self.position
    
    def getFov(self):
        '''视场角'''
        #see {@link http://www.bobatkins.com/photography/technical/field_of_view.html} */
        vExtentSlopeV = 0.5 * self.SensorHeight / self.focalLength
        self.fovVertical = np.rad2deg(2 * np.arctan( vExtentSlopeV ))
        vExtentSlopeH = 0.5 * self.SensorWidth / self.focalLength
        self.fovHorizonal = np.rad2deg(2 * np.arctan( vExtentSlopeH ))
        #print("fov",self.fovVertical,self.fovHorizonal)

    def getIntrinsic(self):
        if self.focalLength==None:
            raise ValueError("focalLength doesn't set")
        if self.ResX==None:
            raise ValueError("ResX doesn't set")
        if self.ResY==None:
            raise ValueError("ResY doesn't set")
        if self.SensorWidth==None:
            raise ValueError("SensorWidth doesn't set")
        if self.SensorHeight==None:
            raise ValueError("SensorHeight doesn't set")
        if self.s==None:
            self.s=0

        self.dx=self.SensorWidth/self.ResX
        self.dy=self.SensorHeight/self.ResY
        self.cx=self.ResX/2
        self.cy=self.ResY/2
        self.M1=np.array([[1/self.dx,0,0,0],[0,1/self.dy,0,0],[0,0,1,0],[0,0,0,1]])
        self.M2=np.array([[self.focalLength,self.s,self.cx,0],[0,self.focalLength,self.cy,0],[0,0,1,0],[0,0,0,1]])
        self.K = self.M2.dot(self.M1) #内参
    
    def getRT(self):
        #外参
        self.getIntrinsic()
        LookAt=self.LookAt
        Position=self.position
        self.Orient=[LookAt[0]-Position[0],LookAt[1]-Position[1],LookAt[2]-Position[2]]
        Rot=getRotationBetweenVector(self.Orient,DEFAULT_LOOKDIRECTION)
        self.Rot=Rot
        self.RT=Rot
        self.RT[0,3]=self.position[0]
        self.RT[1,3]=self.position[1]
        self.RT[2,3]=self.position[2]
        
    def translate(self,tx=0,ty=0,tz=0):
        '''平移操作'''
        self.position[0]+=tx
        self.position[1]+=ty
        self.position[2]+=tz
        self.getRT()

    def rotateOnAxis(self,angle):
        '''绕自身轴旋转'''
        axis=self.getOrient()
        Rot=getRotationBetweenVector(self.Orient,DEFAULT_LOOKDIRECTION)
        rot2=rotateOnAxis(angle,axis)
        self.Rot=Rot.dot(rot2)
        self.getRT()
       
    def plotFrustum(self,ax=None,color="red"):
        '''相机模型'''
        if ax==None:
            # Create a plot
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
        self.getFov()
        self.getRT()
        frustumDepth=self.frustumDepth #视深
        cameraWidth=2*np.tan(np.deg2rad(self.fovHorizonal/2))*frustumDepth
        cameraHeight=2*np.tan(np.deg2rad(self.fovVertical/2))*frustumDepth
        self.cameraHeight=cameraHeight
        self.cameraWidth=cameraWidth
        position=self.position
        direction=normalize(self.Orient)
        center=position+direction*frustumDepth
        z=center[2]
        x=cameraWidth/2
        y=cameraHeight/2
        #leftUp
        OP1=[center[0]-x,center[1]+y,z,1]
        #rightUp
        OP2=[center[0]+x,center[1]+y,z,1]
        #rightBottom
        OP3=[center[0]+x,center[1]-y,z,1]
        #leftBottom
        OP4=[center[0]-x,center[1]-y,z,1]
        plotPoint(center,ax,color="green")
        plotLine(position,OP1,ax,color=color)
        plotLine(position,OP2,ax,color=color)
        plotLine(position,OP3,ax,color=color)
        plotLine(position,OP4,ax,color=color)
        plotLine(OP1,OP2,ax,color="green") # Camera Top
        plotLine(OP2,OP3,ax,color=color)
        plotLine(OP3,OP4,ax,color=color)
        plotLine(OP4,OP1,ax,color=color)
    
    
    def __drawBound__(self):
        bound=self.__ImageBound__()
        self.projectImageDrawBound=True
        for i in range(4):
            q1=bound[i]
            q2=bound[(i+1)%4]
            self.img_ax.plot([q1[0],q2[0]],[q1[1],q2[1]])

    def showOnImage(self,U=0,V=0,color="b"):
        '''在图片上绘制'''
        if img_ax==None:
            fig2 = plt.figure(2)
            self.img_ax = fig2.add_subplot(111)
        #ResX,ResY=self.ResX,self.ResY
        if not self.projectImageDrawBound:
            self.__drawBound__()
        if self.checkPointInBound([U,V],0,0,self.ResX,self.ResY):
            '''只绘制在图像范围内的点'''
            self.img_ax.scatter([U],[V],color=color) 
            self.img_ax.axis("equal")

    def __checkLineIntersectionPoint__(self,L1p1,L1p2,L2p1,L2p2):
        '''计算线段的交点'''
        return checkLineIntersectionPoint(L1p1,L1p2,L2p1,L2p2)
        
    def checkPointInBound(self,p,xmin,ymin,xmax,ymax):
        '''检查点是否在区域内'''
        px,py=p[0],p[1]
        if (xmin<=px and px<=xmax) and (ymin<=py and py<=ymax):
            return True
        else:
            return False
        
    def __ImageBound__(self):
        bound=[[0,0],[0,self.ResY],[self.ResX,self.ResY],[self.ResX,0]]
        return bound
    
    def LineProjectToImage2(self,Pw1,Pw2,targetSize,savePath=None,thickness=1):
        '''计算直线段投影'''
        Pw1=np.array(Pw1)
        Pw2=np.array(Pw2)
        direction_vector=Pw2-Pw1
        length=np.linalg.norm(direction_vector)
        direction=normalize(direction_vector)
        Pc1=self.project2Image(Pw1)
        Pc2=self.project2Image(Pw2)

        camera_position=self.getPosition()
        camera_to_segment_vector=(Pw2-Pw1)/2-camera_position
        d=np.linalg.norm(camera_to_segment_vector)
        p1=self.project2World(0,0,d)
        p2=self.project2World(self.ResX,0,d)
        #p3=self.project2World(self.ResX,self.ResY,d)
        p4=self.project2World(0,self.ResY,d)
        dx=np.linalg.norm(p2-p1)
        dy=np.linalg.norm(p4-p1)
        dx=self.ResX/dx
        dy=self.ResY/dy
        n=np.max(dx*length,dy*length)
        for t in range(100):
            point_on_line=Pw1+t*direction*length
            uv=self.project2Image(point_on_line)


    def LineProjectToImage(self,line:Line,savePath=None,thickness=10):
        '''TODO线投影到图像'''
        line_direction=line.direction
        p0=np.array(p0)
        #计算直线与相机轴线的最近点
        camera_position=np.array(self.position)
        camera_to_line=np.linalg.norm(camera_position-p0)
        closest_point_on_line=p0 + camera_to_line * line_direction

        Pw1=closest_point_on_line+line_direction*10000 #取一个非常大的数
        Pw2=closest_point_on_line-line_direction*10000
        line_segment=self.getIntersectionLineSegment(Pw1,Pw2)
        if line_segment:
            q1=line_segment[0]
            q2=line_segment[1]
            ax.plot([q1[0],q2[0]],[q1[1],q2[1]])
            if savePath:
                saveLineImage(self.ResX,self.ResY,line_segment,255,savePath,thickness)
    def getIntersectionLineSegmentInBound(self,p1,p2,bound=None):
        '''计算线段与边界的交点'''
        if len(p1)==3:
            p1=[p1[0],p1[1]]
        if len(p2)==3:
            p2=[p2[0],p2[1]]
        if bound is None:
            bound=self.__ImageBound__()
        return getILSInBound(p1,p2,bound)

    def getIntersectionLineSegment(self,Pw1,Pw2,bound=None):
        '''计算交线投影后的线段'''
        p1=self.project2Image(Pw1)
        p2=self.project2Image(Pw2)
        if bound is None:
            bound=self.__ImageBound__()
        return getILSInBound(p1,p2,bound)
    
    def CaptureLine(self,Pw1,Pw2,savePath=None,thickness=1):
        if not self.projectImageDrawBound:
            self.__drawBound__()
        line_segment=self.getIntersectionLineSegment(Pw1,Pw2)
        if line_segment:
            q1=line_segment[0]
            q2=line_segment[1]
            self.img_ax.plot([q1[0],q2[0]],[q1[1],q2[1]])
            if savePath:
                saveLineImage(self.ResX,self.ResY,line_segment,255,savePath,thickness)
            return True
    def CaptureLines(self,Lines,savePath=None,thickness=1):
        if not self.projectImageDrawBound:
            self.__drawBound__()
        segments=[]
        for line in Lines:
            line_segment=self.getIntersectionLineSegment(line[0],line[1])
            if line_segment:
                segments.append(line_segment)
        for ls in segments:
            q1=ls[0]
            q2=ls[1]
            self.img_ax.plot([q1[0],q2[0]],[q1[1],q2[1]])
        if savePath:
            saveLinesToImage(self.ResX,self.ResY,segments,255,savePath,thickness)
        return segments
        
    def CaptureLines2(self,Lines,targetSize,savePath=None,thickness=1):
        if not self.projectImageDrawBound:
            self.__drawBound__()
        segments=[]
        for line in Lines:
            line_segment=self.getIntersectionLineSegment(line[0],line[1])
            if line_segment:
                segments.append(line_segment)
        for ls in segments:
            q1=ls[0]
            q2=ls[1]
            self.img_ax.plot([q1[0],q2[0]],[q1[1],q2[1]])
        if savePath:
            saveLinesToImage2(self.ResX,self.ResY,targetSize,segments,255,savePath,thickness)
        return segments
    
    def CaptureLines3(self,Lines,targetSize,Bound,savePath=None,thickness=1):
        '''含边界控制'''
        if not self.projectImageDrawBound:
            self.__drawBound__()
        segments=[]
        for line in Lines:
            line_segment=self.getIntersectionLineSegment(line[0],line[1],Bound)
            if line_segment:
                segments.append(line_segment)
        for ls in segments:
            q1=ls[0]
            q2=ls[1]
            self.img_ax.plot([q1[0],q2[0]],[q1[1],q2[1]])
        if savePath:
            saveLinesToImage2(self.ResX,self.ResY,targetSize,segments,255,savePath,thickness)
        return segments
    def stepZCapture(self,Pw1,Pw2,stepSize=0.2,savePath=None,thickness=1):
        '''相机移动'''
        self.translate(0,0,stepSize)#前进4步，每步0.2m,实际应该是随机的
        self.CaptureLine(Pw1,Pw2,savePath,thickness)

    def project2Image(self,worldCoord=[1,2,3]):
        '''空间点投影到图片'''
        self.getRT()
        # WC=[worldCoord[0],worldCoord[1],worldCoord[2],1]
        # UVAtImage=self.K.dot(self.RT.dot(WC))
        # zc=UVAtImage[2]
        # U=UVAtImage[0]/zc
        # V=UVAtImage[1]/zc
        R=self.Rot[0:3,0:3]
        T=np.array(worldCoord)-np.array(self.position)
        RT=np.dot(R,T)
        K=self.K[0:3,0:3]
        UV=K.dot(RT)
        Zc=UV[2]
        U,V=UV[0]/Zc,UV[1]/Zc
        #图像的Y轴向下为正
        return [U,V] 

    def project2World(self,u,v,Zc):
        '''u,v,(pixel)
           Zc:物距，不是固定的
        '''
        #相机坐标系
        K=self.K[0:3,0:3]
        coord_in_image=[u*Zc,v*Zc,Zc] #逆操作
        coord_in_camera=np.linalg.inv(K).dot(coord_in_image)
        R=self.Rot[0:3,0:3]
        T=[self.position[0],self.position[1],self.position[2]]
        coord_in_world_after_rot=np.linalg.inv(R).dot(coord_in_camera[0:3])
        coord_in_world=coord_in_world_after_rot+T
        
        return coord_in_world
    
    def getOrient(self):
        return [self.LookAt[0]-self.position[0],self.LookAt[1]-self.position[1],self.LookAt[2]-self.position[2]]
   
    def project2Plane(self,u,v,p0,pn):
        K=self.K[0:3,0:3]
        coord_in_image=np.array([u,v,1])
        coord_in_camera=np.linalg.inv(K).dot(coord_in_image)
        R=self.Rot[0:3,0:3]
        pl_direction=np.linalg.inv(R).dot(coord_in_camera)
        pl_origin=np.array(self.position)
        flag,ip=linePlaneIntersection(pl_origin,pl_direction,p0,pn)
        if not flag:
            return None
        else:
            return ip


if __name__=="__main__":
    fig1 = plt.figure(1)
    ax_3d = fig1.add_subplot(111, projection='3d')
    ax_3d.set_xlim(-20,20)
    ax_3d.set_ylim(-20,20)
    ax_3d.set_zlim3d(-20,20)
    ax_3d.set_title("Camera in World")
    ax_3d.view_init(elev=90,azim=-90)
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    fig2 = plt.figure(2)
    img_ax = fig2.add_subplot(111)
    img_ax.set_title("World Point Project on Image/Pixel")
    tunnelHeight=8.5
    tunnelWidth=16.8
    camera=Camera(position=[7.5,1.6,0],LookAt=[7.5,1.6+np.tan(np.deg2rad(20))*13,13],img_ax=img_ax)
    camera.frustumDepth=2 #虚拟相机的视深
    camera.focalLength=20 #调整焦距

    camera.getFov()
    camera.getIntrinsic()
    print("Fov-V:",camera.fovVertical,"Fov-H:",camera.fovHorizonal)
    print("K",np.round(camera.K,2))
    #世界坐标点
    Pw1=[0,0,13]
    Pw2=[15,0,13] #在世界坐标系中Y轴向上，X轴向右
    Pw3=[15,15,13]
    Pw4=[0,15,13] #在世界坐标系中Y轴向上，X轴向右
    plotPoint(Pw1,ax_3d,color="blue")
    plotPoint(Pw2,ax_3d,color="red")
    plotPoint(Pw3,ax_3d,color="yellow")
    plotPoint(Pw4,ax_3d,color="yellow")
    plotLine(Pw1,Pw2,ax_3d,color="g")
    plotLine(Pw3,Pw4,ax_3d,color="b")
    #plotCircelPlane(ax=ax_3d,center=Pw2,normal=[0,0,1],radius=50,color="blue",alpha=0.6)

    #图像中的坐标点
    [U1,V1]=camera.project2Image(Pw1)
    [U2,V2]=camera.project2Image(Pw2)#注意图像坐标系Y向下为正
    [U3,V3]=camera.project2Image(Pw3)
    [U4,V4]=camera.project2Image(Pw4)

    camera.showOnImage(U1,V1,"b")
    camera.showOnImage(U2,V2,"r")
    camera.showOnImage(U3,V3,"g")
    camera.showOnImage(U4,V4,"g")

    #绘制相机模型
    camera.plotFrustum(ax=ax_3d)
    #绘制投影点到图像上
    #目标平面
    point_on_tP=[0,0,13]
    normal_of_tP=[0,0,1]
    
    segments=camera.CaptureLines([[Pw1,Pw2],[Pw3,Pw4]],"./line.png",10)
    print(segments)
    Puv1=segments[0][0]
    Puv2=segments[0][1]
    Pw11=camera.project2Plane(Puv1[0],Puv1[1],point_on_tP,normal_of_tP)
    Pw12=camera.project2Plane(Puv2[0],Puv2[1],point_on_tP,normal_of_tP)
    d=np.linalg.norm(Pw12-Pw11)
    print(Pw11,Pw12,d)
    #camera.stepZCapture(Pw1,Pw2,0,"./StepZCapture",10)
    #相机拍摄范围
    
    #projectBack
    # pw1=camera.project2Plane(U2,V2, point_on_tP,normal_of_tP)
    # print("project back:",pw1)
    Pc1=[0,0]
    Pc2=[camera.ResX,0]
    Pc3=[camera.ResX,camera.ResY]
    Pc4=[0,camera.ResY]

    Pw11=camera.project2Plane(*Pc1,point_on_tP,normal_of_tP)
    Pw21=camera.project2Plane(*Pc2,point_on_tP,normal_of_tP)
    Pw31=camera.project2Plane(*Pc3,point_on_tP,normal_of_tP)
    Pw41=camera.project2Plane(*Pc4,point_on_tP,normal_of_tP)
    #print(Pw11,Pw21,Pw31,Pw41)
    #print(np.linalg.norm(Pw11-Pw21),np.linalg.norm(Pw21-Pw31),np.linalg.norm(Pw31-Pw41),np.linalg.norm(Pw41-Pw11))
    plotLine(Pw11,Pw21,ax_3d)
    plotLine(Pw21,Pw31,ax_3d)
    plotLine(Pw31,Pw41,ax_3d)
    plotLine(Pw41,Pw11,ax_3d)
    plt.show()
    

    