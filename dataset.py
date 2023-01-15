import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms

data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
}

hand_type_dict = {
    'L':0,
    'R':1
}

class FingerCountData(Dataset):
    def __init__(self,root:str,is_train:bool) -> None:
        super().__init__()

        self.root = root 
        self.paths = []
        self.counts = []
        self.hand_types = []

        self.transforms = data_transforms['train'] if is_train else data_transforms['test']

        for f in os.listdir(root):
            path = os.path.join(root,f)
            f_noext = os.path.splitext(f)[0]
            count = int(f_noext.split('_')[1][0])
            hand_type = hand_type_dict[str(f_noext.split('_')[1][1])]

            self.paths.append(path)
            self.counts.append(count)
            self.hand_types.append(hand_type)
        
        assert len(self.paths)==len(self.counts)

     
    def __len__(self):
        return len(self.counts)
    
    def __getitem__(self, index):
        path = self.paths[index]
        count = self.counts[index]
        hand_type = self.hand_types[index]
        image = Image.open(path).convert('RGB')
        tensor = self.transforms(image)
        return tensor,count,hand_type

if __name__=='__main__':
    path = './data/test/0a4d7cbc-2522-4e51-968a-1a86d3b7ee19_5L.png'
    f = '0a4d7cbc-2522-4e51-968a-1a86d3b7ee19_5L.png'
    f_noext = os.path.splitext(f)[0]
    # thing = f_noext.split('_')
    count = int(f_noext.split('_')[1][0])
    hand_type = hand_type_dict[str(f_noext.split('_')[1][1])]
    print()