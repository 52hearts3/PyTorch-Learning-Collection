#定义loss函数
#y=wx+b情况
def compute_error_for_line_given_points(b,w,points):
    total_error=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        total_error=total_error+(y-(w*x+b))**2
    return total_error/float(len(points)) #返回的是平均的loss
#定义梯度函数
def step_gradient(b_current,w_current,points,learning_rate):
    b_gradient=0
    w_gradient=0
    N=float(len(points))
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient=b_gradient+2*(((w_current*x)+b_current)-y)/N  #loss函数求导之后的结果，除以N是因为要进行累加N次，所以求平均
        w_gradient=w_gradient+2*x*(((w_current*x)+b_current)-y)/N
    #开始梯度下降求解
    new_b=b_current-(learning_rate*b_gradient)
    new_w=w_current-(learning_rate*w_gradient)
    return [new_b,new_w]
#定义迭代信息
import numpy as np
def gradient_descent_runner(points,starting_b,starting_w,learning_rate,num_iterations):
    b=starting_b
    w=starting_w
    for i in range(num_iterations):
        b,w=step_gradient(b,w,np.array(points),learning_rate)
    return [b,w]
import numpy as np

# 定义x的范围
x = np.linspace(1,40,100)  # 生成从0到9的数组
# 定义线性关系的参数
m = 2  # 斜率
b = 1  # 截距
# 根据线性关系计算y（不含噪音）
y_true = m * x + b
# 定义噪音
# 这里我们使用正态分布噪音，标准差为1（你可以根据需要调整这个值）
noise = np.random.normal(0, 1, size=x.shape)
# 将噪音加到y上
y_noisy = y_true + noise
# 将x和y_noisy组合成矩阵
matrix = np.column_stack((x, y_noisy))
print(matrix)
# 打印结果
def run():
    points=matrix
    learning_rate=0.001
    initial_b=0
    initial_w=0 #设定初始的w和b是0
    num_iterations=1000
    print('开始梯度下降,初始值b{},w{},error{}'.format(initial_b,initial_w,compute_error_for_line_given_points(initial_b,initial_w,points)))
    print('running')
    [b,w]=gradient_descent_runner(points,initial_b,initial_w,learning_rate,num_iterations)
    print('在{}次迭代后,b={},w={},error={}'.format(num_iterations,b,m,compute_error_for_line_given_points(b,m,points)))
run()