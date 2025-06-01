from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import numpy as np
import cv2

label_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
label_dic = {label: i for i, label in enumerate(label_names)}

train_dir = 'C:\\Users\\User\\Documents\\Pytorch\\DR\\Train\\*\\*.png'
test_dir = 'C:\\Users\\User\\Documents\\Pytorch\\DR\\Test\\*\\*.png'

class ClaheMedian(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8), median_kernel_size=5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.median_kernel_size = median_kernel_size

    def __call__(self, image):
        img_cv = np.array(image)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_clahe = clahe.apply(l)

        enhanced_lab = cv2.merge((l_clahe, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        if self.median_kernel_size > 0:
            enhanced_rgb = cv2.medianBlur(enhanced_rgb, self.median_kernel_size)
        
        return Image.fromarray(enhanced_rgb)

train_transform = transforms.Compose([
    ClaheMedian(clip_limit=2.0, tile_grid_size=(8,8), median_kernel_size=5),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.46238312125205994, 0.26547566056251526, 0.11903924494981766],[0.2862863838672638, 0.16006295382976532, 0.09059497714042664])
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    ClaheMedian(clip_limit=2.0, tile_grid_size=(8,8), median_kernel_size=5),
    transforms.ToTensor(),
    transforms.Normalize([0.46238312125205994, 0.26547566056251526, 0.11903924494981766],[0.2862863838672638, 0.16006295382976532, 0.09059497714042664])
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def default_loader(image):
    return Image.open(image).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, img_dir, transform, loader=default_loader):
        super(MyDataset, self).__init__()

        self.transform = transform
        self.loader = loader
        self.list = glob.glob(img_dir)

    def __getitem__(self, index):
        img_path = self.list[index]
        label_name = self.list[index].split('\\')[-2]
        img_label = label_dic[label_name]
        img_data = self.loader(self.list[index])

        basic_transform = transforms.Compose([
            ClaheMedian(clip_limit=2.0, tile_grid_size=(8,8), median_kernel_size=5),
            transforms.ToTensor(),
            transforms.Normalize([0.46238312125205994, 0.26547566056251526, 0.11903924494981766],[0.2862863838672638, 0.16006295382976532, 0.09059497714042664])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.transform:
            img_data = self.transform(img_data)
        # if label_name == 'No_DR':
        #     img_data = self.transform(img_data)
        # else:
        #     img_data = basic_transform(img_data)


        return img_data, img_label
    
    def __len__(self):
        return len(self.list)

train_dataset = MyDataset(train_dir, train_transform, default_loader)
test_dataset = MyDataset(test_dir, test_transform, default_loader)

num_wor = 4
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_wor, pin_memory=True, persistent_workers=True if num_wor > 0 else False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=num_wor, pin_memory=True, persistent_workers=True if num_wor > 0 else False)
