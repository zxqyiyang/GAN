import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

"""1 展示数据"""
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import cv2
import random
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def reshape2rgb(img):
    img = np.squeeze(img).reshape((3,32,32))
    img = cv2.merge([img[0,:,:],img[1,:,:],img[2,:,:]])
    return img

def showimg(data,count=64,pass_id=None):
    """
    data :多维数组 ==> (N,3,32,32)
    count:图像数量
    """
    fig = plt.figure(figsize=(8, count/8))
    fig.suptitle("Pass {}".format(pass_id))
    gs = plt.GridSpec(int(count/8), 8)
    gs.update(wspace=0.05, hspace=0.05)
    range_number = [random.randint(0, len(data)-1) for i in range(count)]
    for i in range(count):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        img = reshape2rgb(data[range_number[i],:,:,:])
        plt.imshow(img)
    plt.show()

data = unpickle("./data/data68/data_batch_2")
data = data[b"data"].reshape((10000, -1, 32, 32))
print(len(data))
showimg(data, 64, pass_id="cifar10_image")


"""2 引用数据"""
"""目的是将 CIFAR10 的数据转为 paddle 训练所用的数据"""
import paddle as paddle
import paddle.fluid as fluid
def cifar10_reader(batch_number):
    def read():
        filename = "./data/data68/data_batch_" + str(batch_number)
        data = unpickle(filename)
        data = data[b"data"].reshape((10000, -1, 32, 32))
        for i in range(len(data)):
           img = data[i,:,:,:]
           yield img
    return read

def data_generate(i):
    img_data = paddle.batch(paddle.reader.shuffle(cifar10_reader(i), 10000), batch_size=128)
    return img_data
"""检查数据是否生成"""
data = data_generate(1)
post = next(data())
print('一个batch图片数据的形状：batch_size =', len(post), ', data_shape =', post[0].shape)

"""3 产生噪声数据"""
def z_reader():
    while True:
        yield np.random.normal(0.0, 1.0, (100, 1, 1)).astype('float32')  #正态分布，正态分布的均值、标准差、参数
z_generator = paddle.batch(z_reader, batch_size=128)
z_tmp = next(z_generator())
# print(np.squeeze(np.array(z_tmp[0])))
print('一个batch噪声z的形状：batch_size =', len(z_tmp), ', data_shape =', z_tmp[0].shape)

#4 定义 生成器 模型
class G(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(G, self).__init__(name_scope)
        name_scope = self.full_name()
        # My_G的代码
        """
        模型流程：2次全连接+1向上采样+1卷积+1向上采样+1卷积
        注意：除最后一次卷积运算后，其余的输出做一次归一化层；其余用 leaky_relu
        """
        self.fc1 = Linear(input_dim=100, output_dim=1024)
        self.bn1 = fluid.dygraph.BatchNorm(num_channels=1024,act='relu')

        self.fc2 = Linear(input_dim=1024,output_dim=128*8*8)
        self.bn2 = fluid.dygraph.BatchNorm(num_channels=128*8*8,act='relu')

        self.conv1 = Conv2D(num_channels=128,num_filters=64,filter_size=5,padding=2)
        self.bn3 = fluid.dygraph.BatchNorm(num_channels=64, act='relu')

        self.conv2 = Conv2D(num_channels=64, num_filters=3, filter_size=5,padding=2, act='tanh')
        
    def forward(self, z):
        # My_G forward的代码
        z = fluid.layers.reshape(z, [-1, 100])
        x = self.fc1(z)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = fluid.layers.reshape(x, [-1, 128,  8, 8]) #将输出转为 128*7*7 的维度大小

        x = fluid.layers.image_resize(x, scale=2)
        x = self.bn3(self.conv1(x))
        x = fluid.layers.image_resize(x, scale=2)
        x = self.conv2(x)
        return x


#5 定义判别器模型
class D(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(D, self).__init__(name_scope)
        name_scope = self.full_name()
        # My_D的代码
        self.conv1 = Conv2D(num_channels=3, num_filters=64, filter_size=3, padding=0, stride=1)
        self.bn1 = fluid.dygraph.BatchNorm(num_channels=64, act='leaky_relu')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

        self.conv2 = Conv2D(num_channels=64,num_filters=128,filter_size=3, padding=0, stride=1)
        self.bn2 = fluid.dygraph.BatchNorm(num_channels=128, act='leaky_relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

        self.fc1 = Linear(input_dim=128*6*6, output_dim=1024)
        self.bn3 = fluid.dygraph.BatchNorm(num_channels=1024, act='leaky_relu')

        self.fc2 = Linear(input_dim=1024, output_dim=1)

    def forward(self, img):
        # My_G forward的代码
        img = img.astype('float32')
        x = fluid.layers.reshape(img, [-1, 3, 32, 32])
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))

        x = fluid.layers.reshape(x, [-1, 128*6*6])
        x = self.bn3(self.fc1(x))
        x = self.fc2(x)
        return x

#6 测试生成网络G和判别网络D
with fluid.dygraph.guard():
    g_tmp = G('G')
    tmp_g = g_tmp(fluid.dygraph.to_variable(np.array(z_tmp))).numpy()
    print('生成器G生成图片数据的形状：', tmp_g.shape)
    plt.imshow(tmp_g[0][0])
    plt.show()
    
    d_tmp = D('D')
    tmp_d = d_tmp(fluid.dygraph.to_variable(tmp_g)).numpy()
    print('判别器D判别生成的图片的概率数据形状：', tmp_d.shape)
    print(max(tmp_d))

# 展示生成器的图片
# 显示图片，构建一个16*n大小(n=batch_size/16)的图片阵列，把预测的图片打印到note中。
import matplotlib.pyplot as plt
import numpy as np
def normalization(data):
    data_max = np.max(data)
    data_min = np.min(data)
    data = (data - data_min)/(data_max - data_min)
    return data
def tranforg(img):
    img = np.squeeze(img).reshape((3,32,32))
    img = cv2.merge([normalization(img[0,:,:]),normalization(img[1,:,:]),normalization(img[2,:,:])])
    return img
def show_image_grid(images, batch_size=64, pass_id=None):
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Pass {}".format(pass_id))
    gs = plt.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)
    range_number = [random.randint(0, len(images)-1) for i in range(64)]
    # for i, image in enumerate(images):
    for i in range(64):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(tranforg(images[i]), cmap='Greys_r')    
    plt.show()
show_image_grid(tmp_g, 128)

#7 开始训练
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear
import numpy as np
import matplotlib.pyplot as plt
def train(epoch_num=1, batch_size=64, use_gpu=True, load_model=False):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # 模型存储路径
        model_path = './output/'
        d = D('D')
        d.train()
        g = G('G')
        g.train()
        # 创建优化方法
        real_d_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=2e-4, parameter_list=d.parameters())
        fake_d_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=2e-4, parameter_list=d.parameters())
        g_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=2e-4, parameter_list=g.parameters())
        
        # 读取上次保存的模型
        if load_model == True:
            g_para, g_opt = fluid.load_dygraph(model_path+'g')
            d_para, d_r_opt = fluid.load_dygraph(model_path+'d_o_r')
            # 上面判别器的参数已经读取到d_para了,此处无需再次读取
            _, d_f_opt = fluid.load_dygraph(model_path+'d_o_f')
            g.load_dict(g_para)
            g_optimizer.set_dict(g_opt)
            d.load_dict(d_para)
            real_d_optimizer.set_dict(d_r_opt)
            fake_d_optimizer.set_dict(d_f_opt)

        iteration_num = 0
        for epoch in range(epoch_num):
            for i in range(1,6,1):
                image_data = data_generate(i)
                for i, real_image in enumerate(image_data()):
                    # 丢弃不满整个batch_size的数据
                    # print(iteration_num)             
                    if(len(real_image) != 128):
                        continue               
                    iteration_num += 1
                    '''
                    判别器d通过最小化输入真实图片时判别器d的输出与真值标签ones的交叉熵损失，来优化判别器的参数，
                    以增加判别器d识别真实图片real_image为真值标签ones的概率。
                    '''
                    # 将MNIST数据集里的图片读入real_image，将真值标签ones用数字1初始化
                    real_image = fluid.dygraph.to_variable(np.array(real_image))
                    ones = fluid.dygraph.to_variable(np.ones([len(real_image), 1]).astype('float32'))
                    # 计算判别器d判断真实图片的概率
                    p_real = d(real_image)
                    # 计算判别真图片为真的损失
                    real_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_real, ones)
                    real_avg_cost = fluid.layers.mean(real_cost)
                    # 反向传播更新判别器d的参数
                    real_avg_cost.backward()
                    real_d_optimizer.minimize(real_avg_cost)
                    d.clear_gradients()
                    
                    '''
                    判别器d通过最小化输入生成器g生成的假图片g(z)时判别器的输出与假值标签zeros的交叉熵损失，
                    来优化判别器d的参数，以增加判别器d识别生成器g生成的假图片g(z)为假值标签zeros的概率。
                    '''
                    # 创建高斯分布的噪声z，将假值标签zeros初始化为0
                    z = next(z_generator())
                    z = fluid.dygraph.to_variable(np.array(z))
                    zeros = fluid.dygraph.to_variable(np.zeros([len(real_image), 1]).astype('float32'))
                    # 判别器d判断生成器g生成的假图片的概率
                    p_fake = d(g(z))
                    # 计算判别生成器g生成的假图片为假的损失
                    fake_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_fake, zeros)
                    fake_avg_cost = fluid.layers.mean(fake_cost)
                    # 反向传播更新判别器d的参数
                    fake_avg_cost.backward()
                    fake_d_optimizer.minimize(fake_avg_cost)
                    d.clear_gradients()

                    '''
                    生成器g通过最小化判别器d判别生成器生成的假图片g(z)为真的概率d(fake)与真值标签ones的交叉熵损失，
                    来优化生成器g的参数，以增加生成器g使判别器d判别其生成的假图片g(z)为真值标签ones的概率。
                    '''
                    # 生成器用输入的高斯噪声z生成假图片
                    fake = g(z)
                    # 计算判别器d判断生成器g生成的假图片的概率
                    p_confused = d(fake)
                    # 使用判别器d判断生成器g生成的假图片的概率与真值ones的交叉熵计算损失
                    g_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_confused, ones)
                    g_avg_cost = fluid.layers.mean(g_cost)
                    # 反向传播更新生成器g的参数
                    g_avg_cost.backward()
                    g_optimizer.minimize(g_avg_cost)
                    g.clear_gradients()
                
                    # 打印输出
                    if(iteration_num % 10000 == 0):
                        print('epoch =', epoch, ', batch =', i, ', real_d_loss =', real_avg_cost.numpy(),
                            ', fake_d_loss =', fake_avg_cost.numpy(), 'g_loss =', g_avg_cost.numpy())
                        show_image_grid(fake.numpy(), 128, epoch)                             
       
        # 存储模型
        fluid.save_dygraph(g.state_dict(), model_path+'g')
        fluid.save_dygraph(g_optimizer.state_dict(), model_path+'g')
        fluid.save_dygraph(d.state_dict(), model_path+'d_o_r')
        fluid.save_dygraph(real_d_optimizer.state_dict(), model_path+'d_o_r')
        fluid.save_dygraph(d.state_dict(), model_path+'d_o_f')
        fluid.save_dygraph(fake_d_optimizer.state_dict(), model_path+'d_o_f')

train(epoch_num=200, batch_size=128, use_gpu=True) # 10
print(1)