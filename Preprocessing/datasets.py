from typing import Optional, Callable

import os
import ast
import json
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset

from torch.utils.data.distributed import DistributedSampler


class CustomDataset(Dataset):
    def __init__(self, annotation_file_path:str, img_dir:str, include_qa: bool=True, n_positives: int=4, n_negatives: int=4, img_size:int=224, img_processor: Optional[Callable]=None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.include_qa = include_qa
        self.df = pd.read_json(annotation_file_path)#.sample(frac=0.05).reset_index(drop=True)
        self.img_transform = transforms.Compose([
            transforms.Resize((224,224,)),
            transforms.PILToTensor()
        ])
        if self.include_qa:
            self.df['QA'] = self.df['QA'].apply(lambda x: self._process_list_of_qa_dicts(x))
            # print(self.df.QA.apply(lambda x: len(x)).value_counts())
            print(f'len before dropping: %d' % len(self.df))
            self.df = self.df.loc[self.df.QA.apply(lambda x: len(x)).ge(n_positives)]
            # print(f'Len after dropping: %d' % len(self.df))
            print('Dataset_size after dropping images with positive samples less than %d: %d' % (n_positives, len(self.df)))


    @staticmethod
    def _process_list_of_qa_dicts(x):
        output = []
        for i, item in enumerate(x):
            img_qas =[]
            if isinstance(item, str):
                temp = ast.literal_eval(item)
                if len(temp) == 0:
                    continue
                else:
                    for q, a in temp.items():
                        img_qas.append([q,a])
            else:
                print(i)
            output.extend(img_qas)
        return output


    def __len__(self)->int:
        return len(self.df)

    def __getitem__(self, idx)->dict:
        img_path = os.path.join(self.img_dir, str(self.df.iloc[idx]['image_id']).rjust(12, '0') + '.jpg')
        img = Image.open(img_path).convert('RGB')
        # img.show()
        # img = read_image(img_path)
        try:
            img = self.img_transform(img)
        except ValueError:
            print(f'Error processing {img_path}')
            raise RuntimeError

        if self.include_qa:
            # create an array from samples
            qa = self.df.iloc[idx]['QA'][0]

            return {'images': img, 'labels': [], 'qa_positives': qa}

        return {'images': img, 'labels': []}
