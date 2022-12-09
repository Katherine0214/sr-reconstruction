from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['/home/zhangc/project/sr-reconstruction/data/train_data'],
                      test_folders=['/home/zhangc/project/sr-reconstruction/data/test_data'],
                      min_size=20,
                      output_folder='/home/zhangc/project/sr-reconstruction/data/')
    
    
