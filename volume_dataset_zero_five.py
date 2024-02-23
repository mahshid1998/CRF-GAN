import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import numpy as np
import glob

class Volume_Dataset(Dataset):

    def __init__(self, data_dir, mode='train', fold=0, num_class=0):
        self.sid_list = []
        self.data_dir = data_dir
        self.num_class = num_class

        self.class_label_dict = dict()
        if self.num_class > 0: # conditional
            FILE = open("class_label.csv", "r")
            FILE.readline() # header
            for myline in FILE.readlines():
                mylist = myline.strip("\n").split(",")
                self.class_label_dict[mylist[0]] = int(mylist[1])
            FILE.close()

        for item in glob.glob(self.data_dir+"*.npy"):
            sid = item.split('/')[-1]
            if self.class_label_dict[sid] == 5 or self.class_label_dict[sid] == 0:
                self.sid_list.append(sid)

        self.sid_list.sort()
        self.sid_list = np.asarray(self.sid_list)

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        train_index, valid_test = list(kf.split(self.sid_list))[fold]
        kf2 = KFold(n_splits=2, shuffle=True, random_state=0)
        test_index, valid_index = list(kf2.split(self.sid_list[valid_test]))[fold]
        print("Fold:", fold)
        if mode == "train":
            self.sid_list = self.sid_list[train_index]
        elif mode == "test":
            self.sid_list = self.sid_list[valid_test][test_index]
        else:
            self.sid_list = self.sid_list[valid_test][valid_index]
        print("Dataset size:", len(self))


    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, idx):
        img = np.load(self.data_dir+self.sid_list[idx])
        #print("ppp")
        # print("mamooli :",self.sid_list[idx])
        class_label = self.class_label_dict.get(self.sid_list[idx], -1) # -1 if no class label
        # if class_label == -1:
        #     print("rid: ", self.sid_list[idx])
        return img[None,:,:,:], class_label
