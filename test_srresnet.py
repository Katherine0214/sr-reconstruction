from utils import *
from torch import nn
from models import SRResNet
import time
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFilter
 
# 测试图像
imgPath = '/home/zhangc/project/sr-reconstruction/data/test_data/face1.jpg'
 
# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 8      # 放大比例
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
 
if __name__ == '__main__':
 
    # 预训练模型
    srresnet_checkpoint = "/home/zhangc/project/sr-reconstruction/results/model/checkpoint_srresnet_face_8.pth"
 
    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srresnet_checkpoint, map_location={'cuda:0': 'cuda:1'})  # 当用gpu=“1”训练并用“1”推理时，需要将map_location删掉
    srresnet = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    srresnet = srresnet.to(device)
    srresnet.load_state_dict(checkpoint['model'])
   
    srresnet.eval()
    model = srresnet
 
    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')
 
    # 双线性上采样    
    Bicubic_img = img.resize((int(img.width * scaling_factor),int(img.height * scaling_factor)),Image.BICUBIC)
    Bicubic_img.save('/home/zhangc/project/sr-reconstruction/results/result_save/Bicubic_8.jpg')
 
    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)
 
    # 记录时间
    start = time.time()
 
    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
 
    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]   
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save('/home/zhangc/project/sr-reconstruction/results/result_save/SRResNet_8.jpg')
    print('用时  {:.3f} 秒'.format(time.time()-start))
    
    