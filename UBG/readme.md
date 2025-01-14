## 说明
本程序基于开源程序修改，检查了源代码中存在的一些错误，做了修改，使用者请自行验证，no warranty！来源论文:《UnblockGen - A Python library for 3D rock mass generation and analysis》

* 在源代码基础上添加了裂隙面属性输出,法向量、中心点坐标、顶点，倾向/倾角（可以使用mlsteronet绘制裂隙倾向/倾角分布结果）
*  添加了椭圆形裂隙面的生成
*  倾向/倾角基于高斯分布的椭圆形裂隙面生成
*  添加获取裂隙面数量、块体数量的函数，实际上程序也有返回相应列表的功能，使用python中的len()也能获取
*  添加获取生成的block的id、顶点、面、边、中心点、几何体Polygon等几何参数的函数
*  添加了生成一个指定参数（法向量、中心点、宽度、高度）的平面
*  计算裂隙面与目标平面的交点get_IntersectLineToPlane，获取目标平面与裂隙面的交线，目前仅使用圆形裂隙面

## 生成结果

程序生成的dfn为vtk格式，可使用paraview进行浏览，当然你也可以自己写一个可视化界面，将源代码编译成库进行使用。

## 扩展

* 如果你熟悉DDA,获取可以基于该程序进行扩展，它已提供了基本的几何信息，如何让块体的"动起来"，需要你的贡献。DDA原理：https://www.researchgate.net/publication/355696579_Discontinuous_Deformation_Analysis_in_Rock_Mechanics_and_Rock_Engineering

* 基于物理引擎库的仿真，例如Bullet(pyBullet)、web端的canonjs等，做一些动画模拟，隧洞、边坡滑塌，期待你的贡献

## 欢迎关注公众号《知岩智隧》，获取更多信息
