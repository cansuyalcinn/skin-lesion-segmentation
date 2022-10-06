from operator import index
import os
import sys; sys.path.insert(0, os.path.abspath("../"))

import collections
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
thispath = Path(__file__).resolve()

datapath = thispath.parent.parent / "data"
md_df_path = thispath.parent.parent / "metadata"

class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [
            dict(
                collections.Counter(items[~np.isnan(items)]).most_common()
            ) for items in self.labels.T
        ]
        return dict(zip(self.lesion_types, counts))

    def __repr__(self):
        print.print(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not self.datapath_class.exists():
            raise Exception(f"{self.datapath_class} could not be found")
        if not self.metadata_path.exists():
            raise Exception(f"{self.metadata_path} could not be found")
        

class SkinLesion_Dataset(Dataset):

    def __init__(
        self, class_task: str = 'binary',
        df_path: Path = md_df_path,
        datapath_class: Path = datapath,
        seed: int = 0,
        partitions: List[str] = ['train', 'val'],
        n_jobs: int = -1,
        crop_fov: bool = True        
    ):
        """
        Constructor of SkinLesion_Dataset class

        Args:
            class_task (str, optional): Classification task, 'binary' or 'three_class'.
                Defaults to 'binary'.
            df_path (Path, optional): Metadata dataframe path. Defaults to md_df_path.
            datapath_class (Path, optional): data folder containing dataset images.
                Defaults to datapath.
            seed (int, optional): Seed to guarantee reproducibility. Defaults to 0.
            partitions (List[str], optional): Selected sets. Defaults to ['train', 'val'].
            n_jobs (int, optional): Number of processes to use in parallel operations.
                Defaults to -1.
            crop_fov (bool, optional): To crop FOV for images or not. Defaults to True
        """
        super(SkinLesion_Dataset, self).__init__()

        self.class_task = class_task
        self.partitions = partitions
        # self.crop_fov = crop_fov
        
        # Set seed and number of cores to use
        self.seed = seed
        np.random.seed(self.seed)
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        self.datapath_class = datapath_class/class_task
        self.metadata_path = df_path/ (class_task + '.csv')
       
        # Load data
        self.check_paths_exist()
        self.md_df = pd.read_csv(self.metadata_path, index_col=0)

        # Filter partition
        self.filter_by_partition()
        self.labels = self.md_df['label'].values

    def filter_by_partition(self):
        """
        Tthis method is called to filter the images according to the predefined
        partitions given with the original dataset
        """
        self.md_df = self.md_df.loc[self.md_df.split.isin(self.partitions), :]
        self. md_df.reset_index(inplace=True, drop=True)
    
    def resize_image(self, img: np.ndarray):
        return cv2.resh
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample['idx'] = idx
        sample['label'] = self.labels[idx]

        # read and save the image
        img_path = Path(self.md_df['path'].iloc[idx])
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        sample['img'] = img

        return sample
    


