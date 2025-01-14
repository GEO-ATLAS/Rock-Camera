import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
CAMERA_DEFAULT_ORIENT=[0,0,1]
def makeSliceImage(lines,w,h,targetSize,thickness=2,savepath=None):
	image=np.zeros((targetSize[1],targetSize[0]),dtype=np.uint8)
	for line in lines:
		p1=line[0]
		p2=line[1]
		p1x,p1y=int(p1[0]/w*targetSize[0]),int(p1[1]/h*targetSize[1])
		p2x,p2y=int(p2[0]/w*targetSize[0]),int(p2[1]/h*targetSize[1])
		#p11=np.array([p1x,p1y])
		#p21=np.array([p2x,p2y])
		#print("slice:",p1x,p1y,np.linalg.norm(np.array(p1)-p2),np.linalg.norm(p11-p21))
		cv2.line(image,[p1x,p1y],[p2x,p2y],255,thickness)
	cv2.imwrite(savepath,image)
# 绕x轴旋转函数
def rotate_x(angle):
    rotation_matrix = np.array([[1, 0, 0, 0],
                                [0, np.cos(angle), -np.sin(angle), 0],
                                [0, np.sin(angle), np.cos(angle), 0],
                                [0, 0, 0, 1]])
    return rotation_matrix

# 绕y轴旋转函数
def rotate_y(angle):
    rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                                [0, 1, 0, 0],
                                [-np.sin(angle), 0, np.cos(angle), 0],
                                [0, 0, 0, 1]])
    return rotation_matrix

# 绕z轴旋转函数
def rotate_z(angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                                [np.sin(angle), np.cos(angle), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    return rotation_matrix

# 平移函数
def translate(tx, ty, tz):
    translation_matrix = np.array([[1, 0, 0, tx],
                                   [0, 1, 0, ty],
                                   [0, 0, 1, tz],
                                   [0, 0, 0, 1]])
    return translation_matrix

# 绕任意轴旋转函数
def rotate_axis(angle, axis):
    axis = axis / np.linalg.norm(axis)  # 将轴向量归一化
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    rotation_matrix = np.array([[t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1],0],
                                [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0],0],
                                [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c,0],
                                [0,0,0,1]])
    return rotation_matrix

def dipDirectionAngleToNormal(dipDirection,dipAngle):
        '''倾向倾角转法向量'''
        kapa = 0
        echis = (np.pi / 2) - dipAngle
        if 0 <= dipDirection < np.pi:
            kapa = dipDirection + np.pi
        if np.pi <= dipDirection <= 2 * np.pi:
            kapa = dipDirection - np.pi    
        pole = np.array([np.cos(kapa) * np.cos(echis),
                            -np.sin(kapa) * np.cos(echis),
                            -np.sin(echis)])
        return pole,kapa
def randomize(v):
    return v*np.random.rand()

def randomLogNormal(mean,sigma):
    return np.random.lognormal(mean, sigma)

def randomExponential(size):
    scale=1/size
    return np.random.exponential(scale)

def normalize(v):
    return v/np.linalg.norm(v)

def DataWriter(fileName,data:list):
    with open(fileName,"a+") as f:
        num=len(data)
        for i in range(num):
            vi=data[i]
            f.write(str(vi)+",")
        f.write("\n")
        f.close()

def saveLineImage(ResX,ResY,Line,value=255,savePath=None,thickness=1):
    image=np.zeros((ResY,ResX),dtype=np.uint8)
    p1=Line[0]
    p2=Line[1]
    p1=(int(p1[0]),int(ResY-p1[1]))
    p2=(int(p2[0]),int(ResY-p2[1]))
    cv2.line(image,p1,p2,value,thickness=thickness)
    if savePath:
        cv2.imwrite(savePath,image)
        
def saveLinesToImage(ResX,ResY,Lines,value=255,savePath=None,thickness=1):
    image=np.zeros((ResY,ResX),dtype=np.uint8)
    for Line in Lines:
        p1=Line[0]
        p2=Line[1]
        p1=(int(p1[0]),int(ResY-p1[1]))
        p2=(int(p2[0]),int(ResY-p2[1]))
        cv2.line(image,p1,p2,value,thickness=thickness)
    cv2.imwrite(savePath,image)
    
def saveLinesToImage2(ResX,ResY,targetSize,Lines,value=255,savePath=None,thickness=1):
    '''注意图片未上下倒转'''
    dx=targetSize[0]
    dy=targetSize[1]
    fx=dx/ResX
    fy=dy/ResY
    image=np.zeros((targetSize[0],targetSize[1]),dtype=np.uint8)
    for Line in Lines:
        p1=Line[0]
        p2=Line[1]
        p1=(int(p1[0]*fx),int(p1[1]*fy))
        p2=(int(p2[0]*fx),int(p2[1]*fy))
        #print("distance2:",p1,p2,np.linalg.norm(np.array(p1)-p2))
        cv2.line(image,p1,p2,value,thickness=thickness)
    cv2.imwrite(savePath,image)
    
def rotateOnAxis(angle, axis):
    axis = normalize(axis)  # 将轴向量归一化
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    rotation_matrix = np.array([[t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1],0],
                                [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0],0],
                                [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c,0],
                                [0,0,0,1]])
    return rotation_matrix

def rotateOnAxis2(angle, axis):
    axis = normalize(axis)  # 将轴向量归一化
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    rotation_matrix = np.array([[t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
                                [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0]],
                                [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c]])
    return rotation_matrix
def calRegionFace(regionMinCorner, regionMaxCorner):
    face_normals = [
        [0, 0, -1],  # z
        [0, 0, 1],   # z
        [1, 0, 0],   # x
        [-1, 0, 0],  # x
        [0, -1, 0],  # y
        [0, 1, 0]    # y
    ]
    face_centers = [
        [(regionMinCorner[0] + regionMaxCorner[0]) / 2, (regionMinCorner[1] + regionMaxCorner[1]) / 2, regionMinCorner[2]],  # Z
        [(regionMinCorner[0] + regionMaxCorner[0]) / 2, (regionMinCorner[1] + regionMaxCorner[1]) / 2, regionMaxCorner[2]],  # Z
        [regionMaxCorner[0], (regionMinCorner[1] + regionMaxCorner[1]) / 2, (regionMinCorner[2] + regionMaxCorner[2]) / 2],  # x
        [regionMinCorner[0], (regionMinCorner[1] + regionMaxCorner[1]) / 2, (regionMinCorner[2] + regionMaxCorner[2]) / 2],  # x
        [(regionMinCorner[0] + regionMaxCorner[0]) / 2, regionMinCorner[1], (regionMinCorner[2] + regionMaxCorner[2]) / 2],  # y
        [(regionMinCorner[0] + regionMaxCorner[0]) / 2, regionMaxCorner[1], (regionMinCorner[2] + regionMaxCorner[2]) / 2]  # y
    ]
    face_widths = [
        regionMaxCorner[0] - regionMinCorner[0],
        regionMaxCorner[0] - regionMinCorner[0],
        regionMaxCorner[2] - regionMinCorner[2],
        regionMaxCorner[2] - regionMinCorner[2],
        regionMaxCorner[2] - regionMinCorner[2],
        regionMaxCorner[2] - regionMinCorner[2]
    ]

    face_heights = [
        regionMaxCorner[1] - regionMinCorner[1],
        regionMaxCorner[1] - regionMinCorner[1],
        regionMaxCorner[1] - regionMinCorner[1],
        regionMaxCorner[1] - regionMinCorner[1],
        regionMaxCorner[0] - regionMinCorner[0],
        regionMaxCorner[0] - regionMinCorner[0]
    ]
    return face_normals, face_centers, face_widths, face_heights


def getRotationBetweenVector(v1, v2):
    if isinstance(v1,list):
        v1=np.array(v1)
    if isinstance(v2,list):
        v2=np.array(v2)
    
    v1 = normalize(v1)  # Normalize the input vectors
    v2 = normalize(v2)
    if np.isclose(np.sum(v1-v2),0):
        return np.eye(4)
    axis = np.cross(v1, v2)  # Calculate the rotation axis
    axis = axis/np.linalg.norm(axis)  # Normalize the axis

    angle = np.arccos(np.dot(v1, v2))  # Calculate the rotation angle

    # Rodrigues' rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    rotation_matrix = np.array([[t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1],0],
                                [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0],0],
                                [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c,0],
                                [0,0,0,1]
                                ])
    return rotation_matrix

def getRotationBetweenVector2(v1, v2):
    '''向量间的旋转矩阵'''
    if isinstance(v1,list):
        v1=np.array(v1)
    if isinstance(v2,list):
        v2=np.array(v2)
    
    v1 = normalize(v1)  # Normalize the input vectors
    v2 = normalize(v2)
    if np.isclose(np.sum(v1-v2),0):
        return np.eye(3)
    axis = np.cross(v1, v2)  # Calculate the rotation axis
    axis = axis/np.linalg.norm(axis)  # Normalize the axis

    angle = np.arccos(np.dot(v1, v2))  # Calculate the rotation angle

    # Rodrigues' rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    rotation_matrix = np.array([[t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
                                [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0]],
                                [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c]])
    return rotation_matrix

def plotVector(vector,ax=None,**kwargs):
    if ax==None:
        # Create a plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot([0,vector[0]],[0,vector[1]],[0,vector[2]],**kwargs)
    return ax

def plotPoint(p,ax=None,**kwargs):
    if ax==None:
        # Create a plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[0],p[1],p[2],**kwargs)
    return ax

def plotLine(v1,v2,ax=None,**kwargs):
    if ax==None:
        # Create a plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot([v1[0],v2[0]],[v1[1],v2[1]],[v1[2],v2[2]],**kwargs)
    return ax

def plotPlane(ax, p0=[0,0,0],pNormal=[0,0,1],width=10,height=10,**kwargs):
    '''绘制平面'''
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    n=normalize(pNormal)
    #u,v on plane
    if n[2] == 0:  # If the z-component of the normalized normal vector is 0, set n to [0, 0, 1]
        u = np.array([0, 0, 1])
    else:
        u=np.array([1,0,-n[0]/n[2]])
    u=normalize(u)
    v = np.cross(n, u)
    p1 = p0 - width/2 * u - height/2 * v
    p2 = p0 - width/2 * u + height/2 * v
    p3 = p0 + width/2 * u + height/2 * v
    p4 = p0 + width/2 * u - height/2 * v
    vertices = [p1, p2, p3, p4]
    #calculate vertices
    vertices = np.array(vertices)
    ax.add_collection3d(Poly3DCollection([vertices], **kwargs))
    return ax

def plotCircelPlane(ax=None,center=[0,0,0],normal=[0,1,0],radius=1,**kwargs):
    '''绘制圆形平面'''
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    n=normalize(normal)
    #u,v on plane
    if n[2] == 0:  # If the z-component of the normalized normal vector is 0, set n to [0, 0, 1]
        u = np.array([0, 0, -1])
    else:
        u=np.array([1,0,-n[0]/n[2]])
    u=normalize(u)
    v = np.cross(n, u)
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = np.linspace(0, radius, 10)
    U, V = np.meshgrid(theta, radius) #global
    X=center[0]+V*u[0]*np.cos(U)+V*v[0]*np.sin(U)
    Y=center[1]+V*u[1]*np.cos(U)+V*v[1]*np.sin(U)
    Z=center[2]+V*u[2]*np.cos(U)+V*v[2]*np.sin(U)
    ax.plot_surface(X,Y,Z, **kwargs)
    return ax

def plotEllipticalPlane(ax=None, center=[0, 0, 0], normal=[0, 0, 1], major_axis=1, minor_axis=0.5, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    n = normalize(normal)
    if n[2] == 0:  # If the z-component of the normalized normal vector is 0, set n to [0, 0, 1]
        u = np.array([0, 0, -1])
    else:
        u=np.array([1,0,-n[0]/n[2]])
    u = normalize(u)
    v = np.cross(u, n)
    
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = np.linspace(0, 1, 10)
    U, V = np.meshgrid(theta, radius)
    
    X = center[0] + major_axis * V * u[0] * np.cos(U) + minor_axis * V * v[0] * np.sin(U)
    Y = center[1] + major_axis * V * u[1] * np.cos(U) + minor_axis * V * v[1] * np.sin(U)
    Z = center[2] + major_axis * V * u[2] * np.cos(U) + minor_axis * V * v[2] * np.sin(U)
    
    ax.plot_surface(X, Y, Z, **kwargs)
    return ax

def planeIntersectPlane(p1,p1Normal,p2,p2Normal):
    '''平面相交，求直线的参数'''
    # Calculate the direction vector of the intersection line
    n1=normalize(p1Normal)
    n2=normalize(p2Normal)
    direction = np.cross(n1, n2)
    # Check if the direction vector is zero (planes are parallel)
    if np.allclose(direction, 0):
        return False,None,None
    
    A = np.array([n1, n2, direction])
    d = np.array([n1.dot(p1), n2.dot(p2), 0.]).reshape(3,1)
    p_inter = np.linalg.solve(A, d).T
    # Calculate the intersection point
    intersection_point = p_inter[0]
    return True,intersection_point, direction

def planeIntersectCircle(pOrigin,pNormal,cOrigin,cNormal,cRadius):
    '''计算空间中的平面与圆相交'''
    flag,intersection_point,direction=planeIntersectPlane(pOrigin,pNormal,cOrigin,cNormal)
    if not flag:
        return False,None,None
    direction=normalize(direction)#归一化
    line_to_center = cOrigin - intersection_point #直线上一点到圆平面中心点的向量
    distance_to_line = np.dot(line_to_center, direction) #direction 已归一化,但包含方向
    closest_point_on_line = intersection_point + distance_to_line * direction #计算交线上到圆心的最近点
    vector_to_circle = cOrigin - closest_point_on_line #圆心到最近点的向量
    vector_to_circle_norm_square = np.dot(vector_to_circle, vector_to_circle)
    cRadius_square = cRadius * cRadius
    # Calculate the intersection points on the circle
    if vector_to_circle_norm_square <= cRadius_square:
        d=np.sqrt(cRadius_square-vector_to_circle_norm_square)
        intersection_point_1 = closest_point_on_line + direction * d
        intersection_point_2 = closest_point_on_line - direction * d
        return True,intersection_point_1, intersection_point_2
        
    return False,None,None

def plotFrustum(ax=None,O=[0,0,0],LookAtPoint=CAMERA_DEFAULT_ORIENT,cameraWidth=800, cameraHeight=600, frustumDepth=100,color="yellow"):
    '''绘制相机模型'''
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    Orient=[LookAtPoint[0]-O[0],LookAtPoint[1]-O[1],LookAtPoint[2]-O[2]]
    rot=getRotationBetweenVector2(CAMERA_DEFAULT_ORIENT,Orient)
    z=O[1]+frustumDepth
    x=cameraWidth/2
    y=cameraHeight/2
    #leftUp
    OP1=[O[0]-x,O[2]+y,z]
    OP1=np.dot(rot,OP1)
    #rightUp
    OP2=[O[0]+x,O[2]+y,z]
    OP2=np.dot(rot,OP2)
    #leftBottom
    OP3=[O[0]-x,O[2]-y,z]
    OP3=np.dot(rot,OP3)
    #rightBottom
    OP4=[O[0]+x,O[2]-y,z]
    OP4=np.dot(rot,OP4)
    center=(OP1+OP4)/2
    plotPoint(center,ax,color="red")
    plotLine(O,OP1,ax,color=color)
    plotLine(O,OP2,ax,color=color)
    plotLine(O,OP3,ax,color=color)
    plotLine(O,OP4,ax,color=color)
    plotLine(OP1,OP2,ax,color=color)
    plotLine(OP2,OP4,ax,color=color)
    plotLine(OP4,OP3,ax,color=color)
    plotLine(OP3,OP1,ax,color=color)

def linePlaneIntersection(pl_origin, pl_direction,p0, pn):
    '''直线与平面相交'''
    # Calculate the denominator of the intersection formula
    denominator = np.dot(pn, pl_direction)
    
    # Check if the line is parallel to the plane
    if np.abs(denominator) < 1e-6:
        return False,None  # Line is parallel to the plane
    
    # Calculate the numerator of the intersection formula
    numerator = np.dot(pn, (p0 - pl_origin))
    # Calculate the intersection point
    t = numerator / denominator
    intersection_point = np.array(pl_origin) + t * np.array(pl_direction)
    
    return True,intersection_point
def checkLineIntersectionPoint(L1p1,L1p2,L2p1,L2p2):
        '''计算线段的交点'''
        # Convert the points to numpy arrays for easier calculations
        p1 = np.array(L1p1)
        p2 = np.array(L1p2)
        q1 = np.array(L2p1)
        q2 = np.array(L2p2)
        # Calculate direction vectors of the lines
        dir1 = p2 - p1
        dir2 = q2 - q1
        # Calculate determinant to check if lines are parallel
        det = np.cross(dir1[:2], dir2[:2])
        if det == 0:
            # Lines are parallel or coincident
            return False,None
        # Calculate the parameters of the intersection point along each line
        t1 = np.cross(q1 - p1, dir2) / det
        t2 = np.cross(q1 - p1, dir1) / det
        # Check if the intersection point is within the line segments
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            intersection_point = p1 + t1 * dir1
            return True,intersection_point
        else:
            return False,None

def getILSInBound(p1,p2,bound):
        '''计算线段与边界的交点'''
        if len(p1)==3:
            p1=[p1[0],p1[1]]
        if len(p2)==3:
            p2=[p2[0],p2[1]]
        intersections=[]
        if all(p1 >= np.min(bound, axis=0)) and all(p1 <= np.max(bound, axis=0)):
            intersections.append(p1)
    
        if all(p2 >= np.min(bound, axis=0)) and all(p2 <= np.max(bound, axis=0)):
            intersections.append(p2)

        for i in range(4):
            q1=bound[i]
            q2=bound[(i+1)%4]
            flag,intersection=checkLineIntersectionPoint(p1,p2,q1,q2)
            if flag:
                intersections.append(intersection)
        if len(intersections)<2:
            return None
        intersections=[np.array(p) for p in intersections]
        intersections.sort(key= lambda point:np.linalg.norm(point-p1))
        return intersections
if __name__=="__main__":
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Define vertices of the quad (four corners)
    vertices = [
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1]
    ]    
    #圆形平面
    cOrigin=[0,1,-1]
    cNormal=[5,5,5]
    cRadius=5
    plotPoint(cOrigin,ax,color="g")
    plotLine(cOrigin,cOrigin+normalize(cNormal)*10,ax,color="r")
    plotCircelPlane(ax=ax,center=cOrigin,normal=cNormal,radius=cRadius,color="b",alpha=0.6)
    major_axis = 5.5
    minor_axis = 3
    eCenter = [0, 0, 0]
    eNormal=[1,1,1]
    plotEllipticalPlane(ax, eCenter, eNormal, major_axis, minor_axis, color='blue', alpha=0.5)
    #普通平面
    pOrigin=[0.4,0.5,0.1]
    pNormal=[6,3,5]
    plotPlane(ax, pOrigin,pNormal,15,15,facecolors='r', alpha=0.3)
    #计算交线
    flag,p11,p21=planeIntersectCircle(pOrigin,pNormal,cOrigin,cNormal,cRadius)
    print("pp",flag,p11,p21)
    if flag:
        plotPoint(p11,ax,color='r')
        plotPoint(p21,ax,color='r')
        plotLine(p11,p21,ax,color="g")
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim3d(-10,10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()