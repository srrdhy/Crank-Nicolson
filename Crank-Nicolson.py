import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)  #numpy截断数值，显示内容保留3位小数，并四舍五入

L = 1.0
J = 100
dx = float(L)/float(J-1)
x_grid = np.array([j*dx for j in range(J)])

# print(x_grid)

T = 200
N = 1000
dt = float(T)/float(N-1)
t_grid = np.array([n*dt for n in range(N)])

#参数
D_v = float(10.0)/float(100.0)
D_u = 0.01 * D_v

k0 = 0.067
f = lambda u, v: dt*(v*(k0 + float(u*u)/float(1. + u*u)) - u)
g = lambda u, v: -f(u,v)
 
sigma_u = float(D_u*dt)/float((2*dx*dx))
sigma_v = float(D_v*dt)/float((2*dx*dx))

total_protein = 2.26 #蛋白质总量

#初始条件
no_high = 10
U = np.array([0.1 for i in range(no_high,J)] + [2.0 for i in range(0,no_high)])
V = np.array([float(total_protein-dx*sum(U))/float(J*dx) for i in range(0,J)])

# print(U)
# print(V)

#绘制初始条件
'''
plt.plot(x_grid, U)
plt.plot(x_grid, V)
plt.ylim((0., 2.1))
plt.xlabel('x')
plt.ylabel('concentration')
plt.show()
'''

#构建矩阵A、B，都是三对角矩阵，容易构建

A_u = np.diagflat([-sigma_u for i in range(J-1)], -1) +\
      np.diagflat([1.+sigma_u]+[1.+2.*sigma_u for i in range(J-2)]+[1.+sigma_u]) +\
      np.diagflat([-sigma_u for i in range(J-1)], 1)
        
B_u = np.diagflat([sigma_u for i in range(J-1)], -1) +\
      np.diagflat([1.-sigma_u]+[1.-2.*sigma_u for i in range(J-2)]+[1.-sigma_u]) +\
      np.diagflat([sigma_u for i in range(J-1)], 1)
        
A_v = np.diagflat([-sigma_v for i in range(J-1)], -1) +\
      np.diagflat([1.+sigma_v]+[1.+2.*sigma_v for i in range(J-2)]+[1.+sigma_v]) +\
      np.diagflat([-sigma_v for i in range(J-1)], 1)
        
B_v = np.diagflat([sigma_v for i in range(J-1)], -1) +\
      np.diagflat([1.-sigma_v]+[1.-2.*sigma_v for i in range(J-2)]+[1.-sigma_v]) +\
      np.diagflat([sigma_v for i in range(J-1)], 1)

# print(A_u)

#迭代求解 要将我们的系统推进一个时间步长，我们需要先进行一次矩阵向量乘法，然后在右侧进行一次向量向量加法

f_vec = lambda U, V: np.multiply(dt, np.subtract(np.multiply(V, 
                     np.add(k0, np.divide(np.multiply(U,U), np.add(1., np.multiply(U,U))))), U))

# print(f(U[0], V[0]))
# print(f_vec(U, V))

U_record = [] #记录每个时间步长
V_record = []

U_record.append(U)
V_record.append(V)

#用了np.linalg.solve
for i in range(1,N):
    U_new = np.linalg.solve(A_u, B_u.dot(U) + f_vec(U,V))
    V_new = np.linalg.solve(A_v, B_v.dot(V) - f_vec(U,V))
    
    U = U_new
    V = V_new
    
    U_record.append(U)
    V_record.append(V)

#N个时间步长后的结果
'''
plt.plot(x_grid, U)
plt.plot(x_grid, V)
plt.ylim((0., 2.1))
plt.xlabel('x')
plt.ylabel('concentration')
plt.show()
'''

U_record = np.array(U_record)
V_record = np.array(V_record)

fig, ax = plt.subplots()
plt.xlabel('x')
plt.ylabel('t')
heatmap = ax.pcolor(x_grid, t_grid, U_record, vmin=0., vmax=1.2,shading='auto' )
plt.show()