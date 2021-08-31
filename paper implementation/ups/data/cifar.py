import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from .augmentations import RandAugment, CutoutRandom
import pickle
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


##################### TODO : 이미지 어떻게 들어가는지 한번만 확인해보기
def get_imagenet(root='data/datasets', n_lbl=50, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt=''):
    os.makedirs(root, exist_ok=True) #create the root directory for saving data
    # augmentations
    transform_train = transforms.Compose([
        RandAugment(3,4),  #from https://arxiv.org/pdf/1909.13719.pdf. For CIFAR-10 M=3, N=4
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        CutoutRandom(n_holes=1, length=16, random=True)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])

    if ssl_idx is None:
        base_dataset = datasets.ImageNet(root, train=True, download=True)
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl, 10)
        
        os.makedirs('data/splits', exist_ok=True)
        f = open(os.path.join('data/splits', f'cifar10_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']

    lbl_idx = train_lbl_idx
    if pseudo_lbl is not None:
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        #balance the labeled and unlabeled data 
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)
    else:
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None

    train_lbl_dataset = ImageNetAllSSL(
        root, lbl_idx, train=True, transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)
    
    if nl_idx is not None:
        train_nl_dataset = ImageNetAllSSL(
            root, np.array(nl_idx), train=True, transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = ImageNetAllSSL(
    root, train_unlbl_idx, train=True, transform=transform_val)

    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=True)

    if nl_idx is not None:
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, test_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, test_dataset


def lbl_unlbl_split(lbls, n_lbl, n_class):
    lbl_per_class = n_lbl // n_class
    lbls = np.array(lbls)
    lbl_idx = []
    unlbl_idx = []
    for i in range(n_class):
        idx = np.where(lbls == i)[0]
        np.random.shuffle(idx)
        lbl_idx.extend(idx[:lbl_per_class])
        unlbl_idx.extend(idx[lbl_per_class:])
    return lbl_idx, unlbl_idx


class ImageNetAllSSL(datasets.ImageNet):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
        self.targets = np.array(self.targets)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))
        
        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index], self.nl_mask[index]




##################### imagenet : person class에 대해서 필터링해서 가져오는 부분
def get_imagenet_person(root='dataset/', n_lbl=1000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt=''):
    os.makedirs(root, exist_ok=True) #create the root directory for saving data
    # TODO  augmentations 변경, reshape 신경쓰기
    ## TODO : dict로 변경 -> folder, loader
    data_transforms = {
        'Training' : transforms.Compose([
#         RandAugment(3,4), 
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
#         transforms.ColorJitter(
#             brightness=0.4,
#             contrast=0.4,
#             saturation=0.4,
#         ),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
#         CutoutRandom(n_holes=1, length=16, random=True)
            # resizing
    #       transforms.Resize(256), # option 1 
    #       transforms.CenterCrop(256), # option 2 
            transforms.Resize((256,256), interpolation=Image.NEAREST), # option 3
    #       transforms.RandomResizedCrop(256),

            # augmentations
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
            
        ]), 
        'test' :transforms.Compose([
        transforms.Resize((256,256), interpolation=Image.NEAREST), # option 3
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
        ])  
    }
    transform_train = data_transforms['Training'] 
    transform_val =data_transforms['test']
    
    phases = ['Training', 'test']
    image_datasets = {x: datasets.ImageFolder(os.path.join(root, x), data_transforms[x]) for x in phases}
    
    dataloaders = {}
    
    dataloaders['Training'] = DataLoader(image_datasets['Training'], batch_size=4, shuffle=True)#, num_workers=4)
    dataloaders['test'] = DataLoader(image_datasets['test'], batch_size = 4, shuffle=False) #num_workers=4) # batch_size = 4, 200, 500
    
    class_names = image_datasets['Training'].classes
    test_class_names = image_datasets['test'].classes
    
    print('check train classes : ', class_names)
    print('check test classes :', test_class_names)

    if ssl_idx is None:
        ## TODO 
        #base_dataset = datasets.CIFAR10(root, train=True, download=True) # 이 부분이 loader에 들어가기 전 dataset 형태면 됨
        train_root = 'dataset/Training'
        #print('==> check train root : ', train_root)
        base_dataset = image_datasets['Training']
        dataloader = dataloaders['Training']
        
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl, 200)
        print('===> check targets shape :', len(base_dataset.targets))
        
        os.makedirs('data/splits', exist_ok=True)
        f = open(os.path.join('data/splits', f'imagenet_person_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']

    lbl_idx = train_lbl_idx
    
    if pseudo_lbl is not None:
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        #balance the labeled and unlabeled data 
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)
    else:
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None
    
    ## TODO : SSL 함수 변형필요 
    ## loader 먼저 정의 + 뒤에 DatasetFolder 만드는 작업 필요 
    #beta_root = 'data/datasets/imagenet_person/images/Training/person'
    train_lbl_dataset = ImageNetSSL(
        train_root, lbl_idx,loader = dataloaders['Training'], datafolder = image_datasets, 
        transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)
    
    if nl_idx is not None:
        train_nl_dataset = ImageNetSSL(
            train_root, np.array(nl_idx),transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = ImageNetSSL(train_root, train_unlbl_idx, loader = dataloader, datafolder = image_datasets, transform=transform_val) 
    
    print('==> check train_unlbl_indx.shape : ', np.array(train_unlbl_idx).shape) # 1176
    print('==> chekc train_nl_index.shape :', np.array(lbl_idx).shape) # 1000
    
    ## TODO : testset 따로 만들어서 지정해줘야 함 
    test_root = 'dataset/test'
    test_dataset = image_datasets['test']
    test_dataloader = dataloaders['test']

    if nl_idx is not None:
        ## TODO 이건 나중에 변경해도 됨
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, test_dataset#, dataloader
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, test_dataset


    
    
    
    
def lbl_unlbl_split(lbls, n_lbl, n_class):
    
    ## base_dataset.targets, n_lbl, 2
    ## TODO 
#     print('==> check n_lbl :', n_lbl)
#     print('==> check n_class :', n_class)
    
    lbl_per_class = n_lbl // n_class
    
    print('==> check lbl_per_class :', lbl_per_class) 
    
    lbls = np.array(lbls)
    #print('>>> check lbs :', lbls)
    print('check lbls.shape :',lbls.shape )
    
    lbl_idx = []
    unlbl_idx = []
    
    for i in range(n_class):
#         print(np.where(lbls == i))
        idx = np.where(lbls == i)[0]
        print('==> check idx shape : ', idx.shape)
        np.random.shuffle(idx)
        lbl_idx.extend(idx[:lbl_per_class])
        unlbl_idx.extend(idx[lbl_per_class:])
    
    print('>>> check lbl_indx, unlbl_indx :', len(lbl_idx), len(unlbl_idx))
    return lbl_idx, unlbl_idx







class ImageNetSSL(datasets.ImageFolder):
    # unlabeled set, labeled set 자체를 torch dataset 형태로 바꿔주기 위해 상속받아서 사용

    def __init__(self, root, indexs,loader, datafolder, 
                 transform=None, target_transform=None, 
                 pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        
        ## TODO
        super().__init__(root = root,
                         transform=transform,
                         target_transform=target_transform, 
                         loader = loader)
        
        # data -> np.array (이 부분 다시 수정) ##### TODO
        #print('==> check root :', root)
        data = datafolder['Training'] #datasets.ImageFolder(root, transform=transform)
        self.targets = data.targets
        
        self.data: Any = []
        self.targets = []

        for img, target in loader :
            self.data.append(img)
            self.targets.extend(target)
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1)) 
#         print('>>> check data: ', self.data.shape)
        
        #self.data = np.array(loader.dataset)
        #self.data = np.array(data.samples)

#         print('==> check classes :', data.classes)
        
        #self.targets = np.array(self.targets)
        #print('>>> check targets :', self.targets.shape)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))
        
        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            ## TODO
            #print('===> check indexes : ', indexs)
            indexs = np.array(indexs)
#             print('==> check data shape :', self.data.shape)
#             print('==> check target shape :', np.array(self.targets).shape)
#             print('===> check indexes : ', indexs)
#             print('===> check indexes shape : ', indexs.shape)
            #data = np.array(data)
            ## TODO : indexs bound 조정필요
#             print('>> check data :', self.data)
#             self.data = self.data[indexs]
#             self.targets = np.array(self.targets)[indexs]
#             self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
#             print('==> check indexs :', self.indexs)
        else:
            self.indexs = np.arange(len(self.targets))
        

    def __getitem__(self, index):
        
        ## index from indexes
#         print('==> check targets :', np.array(self.targets).shape)
        
#         print('==> check index (default input) :', index)
#         if index < 989 : 
        img, target = self.data[index], self.targets[index]

        ## TODO
        img = Image.fromarray((img*255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        ## TODO : train에서 loader 부분쪽에서 enum으로 받을때 인자로 받아지는 부분
        return img, target, index, self.nl_mask[index] # self.indexs[index] (3)


