# 尝试搭建topo数据并进行简单运算
import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline
#
# fig = plt.figure()
# # ax = fig.add_subplot(111,projection = '3d') 这种形式的写法直接取消，用下面将其转变为在3D绘制图形
# ax = Axes3D(fig)
# # 生成网格数据
# # x = np.arange(-1,4,0.25);y = np.arange(-4,4,0.25)
# # X,Y = np.meshgrid(x,y)
# X,Y,Z = axes3d.get_test_data(0.05)
# # rstride:行之间的跨度；cstride:列之间的跨度
# # ax.plot_wireframe(X,Y,Z,rstride = 10,cstride = 10) # 绘制线框曲线
# # cmap 是颜色映射表
# from matplotlib import cm
# ax.plot_surface(X,Y,Z,rstride = 10,cstride = 10) # 绘制曲面 cmap = cm.rainbow
# # 绘制底部轮廓曲线
# ax.contour(X,Y,Z,zdir='z',offset = -100,cmap = plt.get_cmap('rainbow'))
# plt.show()

# -------------------------------------------------------
# 搭建topo框架（只是简单绘制一个八面体），目前程序设计是通过输入底顶面的各自的四个点实现几何体的输出，这里不考虑任何算法（只是为了辅助理解topo结构）
# 这里的拓扑结构目前只是一个简单线框模型

# 改进 在线框模型的基础上将其更改为表面模型
class MyTopo:
    def __init__(self,pList1,pList2):
        self.pList1 = pList1 # [p1,p2,p3,p4]
        self.pList2 = pList2 # [p5,p6,p7,p8]
        self.vertex = pList1
        self.vertex.extend(pList2)
        self.edge = []
        self.face = []
        self.face_dir = [] # 实际上是面组成边的方向，
        self.shell = []
        self.face_geom = []
        self.tol = 1e-20
    # 构建拓扑关系
    def __topoRelationship(self):
        # 这里要不就简单地进行指定edge的顶点组成
        # PS:个人觉得这里在某种复杂情况下可以将做个拓扑去重，避免冗余
        # ** 拓扑边关系
        self.edge.append((1,2))
        self.edge.append((2, 3))
        self.edge.append((3, 4))
        self.edge.append((4, 1))

        self.edge.append((5,6))
        self.edge.append((6,7))
        self.edge.append((7, 8))
        self.edge.append((8,5))

        self.edge.append((1, 5))
        self.edge.append((2, 6))
        self.edge.append((3, 7))
        self.edge.append((4, 8))

        # ** 拓扑面关系
        self.face.append([1,2,3,4])
        self.face.append([5,6,7,8])

        self.face.append([1,10,5,9])
        self.face.append([2,11,6,10])
        self.face.append([3,12,7,11])
        self.face.append([4,9,8,12])

        # ** 面虽然是由边组成，但是边是有一定的次序，这里应该提供一个可以更改边次序的接口以及判断
        dir = [1,1,1,1] # 底顶面
        dir1 = [1,1,-1,-1]

        self.face_dir.append(dir)
        self.face_dir.append(dir)
        self.face_dir.append(dir1)
        self.face_dir.append(dir1)
        self.face_dir.append(dir1)
        self.face_dir.append(dir1)

        self.shell.extend([1,2,3,4,5,6])

    # 构建曲面模型 采用nurbs 这里源码就不直接写了，直接调用geomdl.Bspline构建B样条曲线
    def __createFace(self):
        # 遍历曲面，从曲面确定边，再由边确定点，即可确定控制点，在进行B样条曲线生成
        count = 0
        for faceItem in self.face:
            # 确定生成曲面的控制点
            # 1 确定顶点下标
            v1 = self.edge[faceItem[0] - 1][0] - 1
            v2 = self.edge[faceItem[0] - 1][1] - 1
            state = count <= 1
            v3 = self.edge[faceItem[2] - 1][1 if state else 0] - 1
            v4 = self.edge[faceItem[2] - 1][0 if state else 1] - 1
            control_points = [self.vertex[v1],self.vertex[v2],self.vertex[v3],self.vertex[v4]]
            surf = BSpline.Surface()
            surf.degree_u = 1
            surf.degree_v = 1
            surf.set_ctrlpts(control_points,2,2)
            surf.knotvector_u = [0, 0, 1, 1]
            surf.knotvector_v = [0, 0, 1, 1]
            surf.delta = 0.05
            self.N_u = int(1/surf.delta_u)
            self.N_v = int(1/surf.delta_v)
            self.face_geom.append(surf.evalpts)
            count += 1

    def __getNormal(self,p1,p2,p3):
        vector1 = np.array(self.vertex[p2 - 1]) - np.array(self.vertex[p1 - 1])
        vector2 = np.array(self.vertex[p3 - 1]) - np.array(self.vertex[p1 - 1])
        normal = np.cross(vector1,vector2)
        normal_len = np.linalg.norm(normal)
        return normal/normal_len

    def faceAddEdge(self):
        faceNewAdd = []
        facedirNewAdd = []
        # 1.将每个面的边按我们预定的顺序顺时针确定确定每个面的点顺序
        for faceItem,facedirItem in zip(self.face,self.face_dir):
            facePoint = []
            for edgeItem,edgedir in zip(faceItem,facedirItem):
                facePoint.append(self.edge[edgeItem - 1][0] if edgedir == 1 else self.edge[edgeItem - 1][1])
            # 2.这里简单就行，选择第一个点并按对角进行，平面法向量的判断即可，若存在折叠则需要将增加边，以及增加对应的面
            # 直接三个点确定该三角面片的单位法向量，并与对角面片的单位法向量内积确定是否在同一面片上，若不在同一面片上则直接进行
            normal1 = self.__getNormal(facePoint[0],facePoint[1],facePoint[2])
            normal2 = self.__getNormal(facePoint[3],facePoint[1],facePoint[2])
            isSameFace = np.abs(np.abs(np.dot(normal1,normal2)) - 1) < self.tol

            # 增加对应的面以及增加对应的边，这里不追究面的父面是什么，只是简单地进行将新增加的面与面的边方向加进去
            if ~isSameFace:
                edge = (facePoint[1],facePoint[3])
                self.edge.append(edge)
                edge_index = len(self.edge)
                # 将相关面进行分割，重新设置该面的组成边以及设置面的组成边方向
                edge1 = faceItem[0]
                edge2 = faceItem[1]
                edge3 = faceItem[2]
                edge4 = faceItem[3]
                faceItem = []
                facedirItem = []
                faceNewAdd.append([edge1,edge_index,edge4])
                facedirNewAdd.append([1,1,1])

                faceNewAdd.append([edge2,edge3,edge_index])
                facedirNewAdd.append([1,1,-1])
        self.face.extend(faceNewAdd)
        self.face_dir.extend(facedirNewAdd)


    def plotPoly(self):
        self.__topoRelationship()
        self.__createFace()
        # 需要加个条件判断，判断我们预设的八面体中的每个面是否有进行折叠，若有进行折叠则对其进行数据边的添加
        self.faceAddEdge()
        fig = plt.Figure()
        ax = plt.axes(projection='3d')
#         ax = plt.gca()
        # ax = Axes3D(fig) # 这个有时候使用不了会报错
        # 这里的绘制绘制逻辑应该是从shell开始，这种结构写完就开始搞WK_BIM
        drawEdge = []
        for i in self.shell:
            facegeomItem = self.face_geom[i - 1]
            faceItem = self.face[i - 1]
            facedirItem = self.face_dir[i - 1]
            for edgeItem,edgedirItem in zip(faceItem,facedirItem):
                if edgeItem not in drawEdge:
                    vertex1 = self.vertex[(self.edge[edgeItem - 1][0] if edgedirItem == 1 else self.edge[edgeItem - 1][1]) - 1]
                    vertex2 = self.vertex[(self.edge[edgeItem - 1][1] if edgedirItem == 1 else self.edge[edgeItem - 1][0]) - 1]
                    x = np.array([vertex1[0],vertex2[0]])
                    y = np.array([vertex1[1],vertex2[1]])
                    z = np.array([vertex1[2],vertex2[2]])
                    ax.plot(x,y,z,'red')
                    drawEdge.append(edgeItem)
            # 这里简单地绘制一面曲面看看
            surface_points = facegeomItem
            X = np.array([P[0] for P in surface_points])
            X.resize((self.N_u, self.N_v))
            Y = np.array([P[1] for P in surface_points])
            Y.resize((self.N_u, self.N_v))
            Z = np.array([P[2] for P in surface_points])
            Z.resize((self.N_u, self.N_v))
            ax.plot_surface(X, Y, Z)
        plt.show()

if __name__ == "__main__":
    pList1 = [(0,0,0),(4,0,0),(4,4,0),(0,4,4)]
    pList2 = [(0,0,5),(4,0,5),(4,4,5),(0,4,8)]
    topo = MyTopo(pList1,pList2)
    topo.plotPoly()

