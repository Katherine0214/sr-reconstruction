# sr-reconstruction
>本算法包用于实现图像的超分辨率重建，并提供SRResNet、SRGAN两种超分重建模型。

📖 一、目录文件说明
---
项目根目录下有8个```data```文件和2个文件夹，下面对各个文件和文件夹进行简单说明：

- ```create_data_lists.py```：生成数据列表，检查数据集中的图像文件尺寸，并将符合的图像文件名写入JSON文件列表供后续Pytorch调用；
- ```datasets.py```：用于构建数据集加载器，主要沿用Pytorch标准数据加载器格式进行封装；
- ```models.py```：模型结构文件，存储各个模型的结构定义；
- ```utils.py```：工具函数文件，所有项目中涉及到的一些自定义函数均放置在该文件中；
- ```train_srresnet.py```：用于训练SRResNet算法；
- ```train_srgan.py```：用于训练SRGAN算法；
- ```eval.py```：用于模型评估，主要以计算测试集的PSNR和SSIM为主；
- ```test.py```：用于单张样本测试，运用训练好的模型为单张图像进行超分重建；
- ```data```：用于存放训练和测试数据集以及文件列表；
- ```results```：用于存放运行结果，包括训练好的模型以及单张样本测试结果；

⚡ 二、训练
---
（1）将训练数据集放到```data/train_data```文件夹下；

（2）运行```create_data_lists.py```文件为数据集生成文件列表；

（3）运行```train_srresnet.py```进行SRResNet算法训练，训练结束后在```results/model```文件夹中会生成```checkpoint_srresnet.pth```模型文件。

👀 三、测试
---
（1）将测试数据集放到```data/test_data```文件夹下；

（2）运行```test_srresnet.py```文件对```data/test_data```文件夹下的图像进行超分还原，还原结果存储在```results/result_save```文件夹下。

💎 四、评估
---
（1）将测试数据集放到```data/test_data```文件夹下；

（2）运行```eval_srresnet.py```文件对测试集进行评估，计算每个测试集的平均PSNR、SSIM值。

