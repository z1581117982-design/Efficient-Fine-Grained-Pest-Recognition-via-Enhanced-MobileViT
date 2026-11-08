# 定义VGGNet网络模型
import torch
import torch.nn as nn
from torchsummary import summary


class VGG(nn.Module):  # 括号里面是父类
    # init():进行初始化，申明模型中各层的意义
    # features:生成提取特征的网络结构
    # num_class :需要分类的类别个数  VGG网络的不同之处，只是在特征提取上有所不同之处 VGG16,VGG19只是提取特征不同
    # init_weights：是否对网络进行权重初始化
    def __init__(self, features, num_classes=1000, init_weights=False):
        # super引入父类的初始化方法对子类进行初始化
        super(VGG, self).__init__()  # 调用父类的__init__方法，通常是用于定义神经网络框架的一个类
        # 生成提取特征的网络结构
        self.features = features  # 参数赋值给一个实例对象
        # 生成分类的网络结构
        # sequential:自定义顺序连接成网络，生成网络结构
        self.classifier = nn.Sequential(
            # Dropout:随机地将输入中50%地神经元激活为0，即去掉一些神经节点，防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            # RELU(INPLACE=TRUE):将tensor直接修改，不找变量做中间的传递，节省运算内存，不用多存储额外地变量
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        # 如果为真，则对网络参数进行初始化
        if init_weights:
            self._initialize_weights()
        # forward():定义前向传播过程，描述了各层之间恶的关系

    def forward(self, x):
        # 将数据输入到提取特征地网络结构，N*3*224*224
        x = self.features(x)
        # N*512*512*7
        # 图像经过提取地特征网络之后，进行展平，将多维数组进行平坦化处理，神经网络中第0维是batch_size,所以从第二位开始平坦化
        x = torch.flatten(x, start_dim=1)
        # 将数据输入分类网络结构，N*512*512*7
        x = self.classifier(x)
        return x

    # 网络结构参数初始化
    def _initialize_weights(self):
        # 遍历网络中的每一层
        # 集成nn。moudle类的一个方法是self.moudles(),他会返回该网络中的所有moudles
        for m in self.modules():  # 遍布了模型的所有模块
            # isijnstance(object,type):如果指定对象是指定类型，则instance（）函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # unifrom_(tensor,a=0,b=1):服从U（a,b）均匀分布，进行初始化
                nn.init.xavier_uniform_(m.weight)  # 使用 Xavier 均匀分布初始化当前卷积层的权重。m.weight 表示当前层的权重参数。
                # 如果偏置不是0，将偏置设为0，对偏置进行初始化
                if m.bias is not None:
                    # constant_(tensor,val):初始化整个矩阵为常熟l
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 全连接层一定有w和b
                # 正太分布齿梳化
                nn.init.xavier_uniform_(m.weight)
                # 将所有偏置置为0
                nn.init.constant_(m.bias, 0)


# 生成提取特征的网络结果
# 参数是网络配置变量，传入对应配置的列表（list类型）
def make_features(cfg: list):
    # 定义空列表，用于创建的每一层的结构
    layers = []  # 因为每个层都是pytorch的对象，添加到layers，这个列表的元素就是一个个的pytorch层对象，然后通过nn。sequential（*layers）来构建网络
    # 输入图像是RGB彩色图片
    in_channels = 3
    # for循环遍历配置列表，得到由卷积层和池化层组成的一个列表
    for v in cfg:
        # 如果列表的值是M字符，说明该层是最大池化层
        if v == "M":
            # 创建一个最大池化层，在VGG中所有的最大池化层的kernel_size=2,stride=2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 否则是卷积层
        else:
            # inchannels:输入的特征矩阵的深度，v:输出特征矩阵的深度，深度也就是卷积核的个数
            # 在VGG中，所有卷积层的填充为1，步长为1
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            # 将卷积层和RELU放入列表
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
        # 将列表通过非关键字参数的形式返回，*layers可以接受任意数量的参数
    return nn.Sequential(*layers)


# *是用来解包列表的操作符，这个操作会将列表中的所有元素解包成单独的参数传递nn。sequential

# 定义cfgs字典文件，每一个key代表一个模型的配置文件，如：VGG11代表A配置，也就是一个11层的网络
# 数字代表卷积层中卷积核的个数，'M'代表池化层的结构
# 通过函数make_features(cfg: list)生成提取特征网络结构
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 键是一个字符串，值是一个列表
# 实例化给定的配置模型，这里使用的是VGG16
# **kwargs表示可变长度的字典变量。在调用VGG函数时传入的字典变量
def vgg(model_name="vgg16", **kwargs):  # 第二个参数的意思是接受任意数量的关键字参数，这意味着你可以
    # 如果model_name不在cfgs，序会抛出AssertionError错误，报错为参数内容“ ”
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    # 得到VGG16对应的列表
    cfg = cfgs[model_name]
    # 实例化VGG网络
    # 这个字典变量包含了分类的个数以及是否初始化权重的布尔变量
    model = VGG(make_features(cfg), **kwargs)
    return model

if __name__ == '__main__':
    # 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Model loaded successfully.")
    x = torch.randn(4, 3, 224, 224).to(device)
    model = vgg("vgg16", num_classes=1000).to(device)
    # 模型推理
    with torch.no_grad():  # 关闭梯度计算
        model.eval()  # 设置模型为评估模式
        output = model(x)
    print(f"Output shape: {output.shape}")
    # 打印模型摘要（可选）
    summary(model, input_size=(3, 224, 224))