from unblocks import *
import plotTools as plotTools
import matplotlib.pyplot as plt
import numpy as np
import mplstereonet
#定义裂隙网络模型
dfn = DFN()
dfn.set_RandomSeed(100)#设置随机数-则相同随机数下生成的模型是相同的
dfn.set_RegionMaxCorner([15,15,150])
dfn.add_FractureSet()#添加裂隙集1
dfn.add_FractureSet()#添加裂隙集2
#dfn.add_FractureSet()#添加裂隙集3
dfn.add_LineMapping([10,30,3], [0.766582,4.631392,24.30794]) #线映射
#dfn.add_QuadrilateralMapping([0,0,20],[100,0,20],[100,100,20],[0,100,20])#面映射
dfn.add_VolumeMapping()#体积映射
#生成裂隙集
while (dfn.linesMapping[0].get_P10(0) < 0.04):
	dfn.fractureSets[0].add_BaecherFracture(110,48,50000,"det",200,0)
# while (dfn.surfacesMapping[0].get_P21(0) < 0.2):
# 	dfn.fractureSets[1].add_BaecherFracture(90, 45, 20, "exp", 40, 0);
while (dfn.volumesMapping[0].get_P30(1) < 0.001):
	dfn.fractureSets[1].add_BaecherFracture(70, 45, 100, "exp", 5, 0);
while (dfn.volumesMapping[0].get_P32(1) < 0.002):
	dfn.fractureSets[1].add_BaecherFracture(180, 65, 100, "exp", 4, 1);

#向裂隙面集合1添加椭圆形裂隙面
fracCenter=[7.5,7.5,10]
fracDirection=175
fracdipAngle=90
fracRadius=5
dfn.fractureSets[0].add_EllipseFracture(fracCenter,fracDirection,fracdipAngle,fracRadius,fracRadius*2)
#添加一个倾向/倾角高斯分布的椭圆形裂隙面
meanDipDirection=75
meanDipAngle=60
sigmaDipDirection=5
sigmaDipAngle=3
sizeDistribution='exp' #det 、log
meanFractureSize=5
sigmaFractureSize=2
for i  in range(10):
	dfn.fractureSets[0].add_GaussDistFracture(meanDipDirection, meanDipAngle, sigmaDipDirection, sigmaDipAngle, sizeDistribution, meanFractureSize, sigmaFractureSize)
#导出区域
dfn.export_RegionVtk("./cube/region")
#获取裂隙集数量
print("FractureSetNum:",dfn.get_FractureSetsNum(),dfn.FractureSetsNum)#两个一样
tpCenter=[15,15,75]
tpNormal=[0,0,1]
dipDirectionsAndAngle=[]
#遍历每个生成的裂隙集及其裂隙面
for i in range(dfn.FractureSetsNum):
	fs=dfn.fractureSets[i]
	n=fs.get_FractureNum() #集合中的裂隙数量
	for k in range(n):
		f=fs.fractures[k]#获取裂隙面
		#打印裂隙面的属性
		print("id",f.id,"Center:",f.get_Center(),"normal:",f.get_UnitVector(),"radius:",f.get_BoundingSphereRadius(),"area:",f.get_Area())
		#print("direction/angle",f.get_DipDirectionDipAngle())
		dipDirectionsAndAngle.append(f.get_DipDirectionDipAngle())
		#计算裂隙面与目标平面的交点
		intersection_points=f.get_IntersectLineToPlane(tpCenter,tpNormal)
		print("intersection_points",intersection_points)
dfn.export_DFNVtk("./cube/dfn")
#块体生成
generator = Generator()
#参数设置
generator.set_MinInscribedSphereRadius(0.05) 
generator.set_MaxAspectRatio(30)
#执行生成
generator.generate_RockMass(dfn)
#生成一个平面，输入平面中心点、法向量、宽度、高度---作用可自行探索
generator.generate_PlaneVTK("./cube/plane1",tpCenter,tpNormal,50,50)
#访问块体信息
#块体数量
blocksNum=generator.get_BlocksNum()
for i in range(blocksNum):
	block0=generator.blocks[i]
	print("#"*20,"block-",i,"块体信息","#"*20)
	print("Id:",block0.get_Id(),"Volume:",block0.get_Volume())
	print("Alpha:",block0.get_Alpha(),"Beta:",block0.get_Beta(),"AspectRatio:",block0.get_AspectRatio())
	print("*"*20,"block-",i,"几何信息","*"*20)
	print("Center:",block0.get_Center())
	print("Vetices:",block0.get_Vertices())
	print("Edges:",block0.get_Edges())
	print("Planes:",block0.get_Planes())
	print("Polygons:",block0.get_Polygons())
	block0.export_BlockVtk("./cube/blocks/b"+str(i))#导出单个块体
generator.export_BlocksVtk("./cube/block")
dipDirectionsAndAngle=np.array(dipDirectionsAndAngle)

#绘制块体统计结果
#统计
plotTools.blockVolumeDistribution(generator.get_Volumes(True))
plotTools.blockShapeDiagram(generator.get_AlphaValues(True), generator.get_BetaValues(True), generator.get_Volumes(True), 0.05)
plotTools.BlockShapeDistribution(generator.get_AlphaValues(True), generator.get_BetaValues(True), generator.get_Volumes(True))
plotTools.showPlots()

#Stero统计
fig, ax = mplstereonet.subplots()
cax = ax.density_contourf(dipDirectionsAndAngle[:,0], dipDirectionsAndAngle[:,1], measurement='poles')
ax.pole(dipDirectionsAndAngle[:,0], dipDirectionsAndAngle[:,1])
ax.grid(True)
fig.colorbar(cax)
plt.show()