from VirtualCamera.utils import *
class Line:
    def __init__(self,p0,direction) -> None:
        if len(direction)!=3:
            raise ValueError("direction size must be 3")
        if len(p0)!=3:
            raise ValueError("p0 size must be 3")
        self.p0=p0
        self.direction=normalize(direction)
    
    def checkPointAtLine(self,p):
        d=self.direction.dot(np.array(p)-self.p0)
        return np.isclose(d,0)

    def plot(self,length=10,ax=None,p0=None,**kwargs):
        '''绘制'''

        if p0 is None or len(p0)!=3 or (not self.checkPointAtLine(p0)):
            p0=self.p0
        return plotLine(p0-self.direction*length,p0+self.direction*length,ax=ax,**kwargs)
    
    def getPointAtLength(self,length):
        '''计算p0沿直线length长度位置的坐标'''
        return self.p0+self.direction*length
    
    
    
