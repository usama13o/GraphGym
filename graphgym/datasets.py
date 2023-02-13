from bisect import bisect_right
import datetime
import glob
import math
from os import listdir
from random import sample
import numpy as np
from sklearn.feature_selection import SelectFdr
import tifffile
from torch import dtype
import torch.utils.data as data
from pathlib import Path
# Standard libraries
import torch.utils.data as data
import numpy as np
import pickle 
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
from os.path import join
def open_pickled_file(fn):
  with open(fn, "rb") as f_in:
    arr_new = pickle.load(f_in)
  return arr_new

class Whole_Slide_Bag(data.Dataset):
    def __init__(self,
        file_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size=-1,
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained=pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        # self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
        
        return img


class svs_h5_dataset(data.Dataset):
    def find_bin(self,y):
        l = [0]
        for ll,x in zip(l,self.tot):
            l.append(x+ll)
            where = list(map((lambda x: x-y ),l))
            which = bisect_right(where,0)
        return which
    def __init__(self, root_dir, split="all", transform=None, preload_data=False,train_pct=0.8,balance=True):
        super(svs_h5_dataset,self).__init__()
        #train dir 
        img_dir = root_dir

        self.image_filenames  = sorted([join(img_dir, x) for x in listdir(img_dir) if ".h5" in x ])

        # get total patches in each WSI
        tot=[]
        for can in range(len(self.image_filenames)):
            fn = self.image_filenames[can]
            tot.append(len(Whole_Slide_Bag(fn)))
        self.tot = tot
        
        self.target_filenames = []
        sp= self.image_filenames.__len__()
        sp= int(train_pct *sp)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        elif split =='all':
            self.image_filenames = self.image_filenames
        else:
            self.image_filenames = self.image_filenames[sp:]
            # find the mask for the image
        #assert len(self.image_filenames) == len(self.target_filenames)
        tot=[]
        for can in range(len(self.image_filenames)):
            fn = self.image_filenames[can]
            tot.append(len(Whole_Slide_Bag(fn)))
        self.tot = tot
        # report the number of images in the dataset
        print('Number of {0} images: {1} svs'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [open_image_np(ii)[0] for ii in self.image_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        target = 0 #self.target_filenames[index]
        # get which Image 
        
        where = self.find_bin(index) - 1
        try:
            input = Whole_Slide_Bag(self.image_filenames[where])
        except:
            print(f"Couldnt find Image with index {where} with input index of {index}")

        # Which index in that image         
        which = index & len(input) - 1


        # load the nifti images
        if not self.preload_data:
            try:
                input = input[which]
            except:
                print(f"Couldn't find patch with index {which} in SVS with total of {len(input)} patches")
        else:
            input = np.copy(self.raw_images[index])

        # handle exceptions
        if self.transform:
            input = self.transform(input)

        return input, target

    def __len__(self):
        return sum(self.tot)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz",'png','tiff','jpg',"bmp"])

def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).
    Args:
    filename: Name of the image file.
    returns:
    A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image
def open_image_np(path):
    im = open_image(path)
    array = np.array(im)
    return array


# combine multiple datasets into one Class 
from medmnist.dataset import PathMNIST, BreastMNIST,OCTMNIST,ChestMNIST,PneumoniaMNIST,DermaMNIST,RetinaMNIST,BloodMNIST,TissueMNIST,OrganAMNIST,OrganCMNIST,OrganSMNIST
from medmnist.dataset import MedMNIST2D
class combined_medinst_dataset(MedMNIST2D):
    def __init__(self, root="",split="train",transform=None,no_dataset=11,limit=None):
        # load the datasets
        self.transform =transform
        self.limit = limit
        pathmnist = PathMNIST(split=split, root=root)
        breastmnist = BreastMNIST(split=split,root=root)
        octmnist = OCTMNIST(split=split, root=root)
        chestmnist = ChestMNIST(split=split,root=root)
        pneumoniamnist = PneumoniaMNIST(split=split, root=root)
        dermamnist = DermaMNIST(split=split, root=root)
        retinamnist = RetinaMNIST(split=split, root=root)
        bloodmnist = BloodMNIST(split=split, root=root)
        organA = OrganAMNIST(split=split, root=root)
        organC = OrganCMNIST(split=split, root=root)
        organS = OrganSMNIST(split=split, root=root)
        tissueMnist = TissueMNIST(split=split, root=root,download=True)
        datasets = [pathmnist,breastmnist,octmnist,chestmnist,pneumoniamnist,dermamnist,retinamnist,bloodmnist,organA,organC,organS,tissueMnist]
        self.datasets = datasets
        self.tot  = sum(len(d) for d in self.datasets)
        if limit is not None:
            self.d = []
            ex=0
            for d in self.datasets:
                if int(limit / no_dataset) > len(d):# if the limit is greater than the dataset
                    self.d.append(len(d))
                    self.d[ex] +=(int(limit / no_dataset) - len(d))
                else:
                    self.d.append(int(limit / no_dataset))
            print("Datasets split -- > ",self.d)
        else:
            self.d = [len(d) for d in self.datasets]
        #create e dictionary mapping between the dataset index in datasets and the class index in the combined dataset
        class_index = {}
        for i,d in enumerate(datasets):
            for j in range(len(d.info['label'])):
                class_index[i,j] = sum(len(d.info['label']) for d in datasets[:i]) + j
        self.class_index = class_index
        print(self.class_index)

    def __getitem__(self, i):
        #based on i and total number of samples in all datasets, determine which dataset to get the sample from
        for y,d in enumerate(self.datasets):
            # print("looking in dataset",d," for sample",i)
            if i < self.d[y]:
                # print("Debug ----> ",i,self.d,d)
                x,z = d[i]
                if len(z) > 1:
                    z =  np.array(0) if sum(z) == 0 else np.array(1)
                # if image in index 1 has 1 channel, repeat it 3 times then reutrn it
                if x.mode == 'L':
                    return self.transform(x.convert("RGB")), self.class_index[(y,int(z))]
                return self.transform(x), self.class_index[(y,int(z))]
            i -= self.d[y]
        raise IndexError('index out of range')

    def __len__(self):
        if self.limit is not None:
            return self.limit
        #sum of all the lengths of the datasets
        return self.tot

class ConcatDataset(data.Dataset):
    """Dataset to concatenate multiple datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        for d in self.datasets:
            
            if i < len(d):
                return d[i]
            i -= len(d)
        raise IndexError('index out of range')

    def __len__(self):
        return sum(len(d) for d in self.datasets)


class ImageFolder2(ImageFolder):
    "same as ImageFolder but with limit to the number of classes,assumes equally distributed number of iamges per class"
    def __init__(
            self,
            root: str,
            transform = None,
            target_transform = None,
            is_valid_file= None,
            limit = None,
    ):
        super(ImageFolder2, self).__init__(root,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        # limit the classes in self.samples to the number of classes in self.limit
        if limit is not None:
            self.samples = [s for s in self.samples if s[1] < limit]
            self.targets = [s[1] for s in self.samples]
from torch_geometric.data import Data

import torch_geometric
from torch_geometric.transforms import BaseTransform
from skimage import future
from torch_scatter import scatter_min
from torch_scatter import scatter_mean
from torch_geometric.data import Data
from skimage import graph, data, io, segmentation, color
# tensor prep
import torch
from torch_geometric.utils import grid
import torchvision.transforms as T
from math import sqrt
class ImgToGraph(BaseTransform):
    r"""Converts an image to a superpixel representation using the
    :meth:`skimage.segmentation.slic` algorithm, resulting in a
    :obj:`torch_geometric.data.Data` object holding the centroids of
    superpixels in :obj:`pos` and their mean color in :obj:`x`
    (functional name: :obj:`to_slic`).

    This transform can be used with any :obj:`torchvision` dataset.

    Example::

        from torchvision.datasets import MNIST
        import torchvision.transforms as T
        from torch_geometric.transforms import ToSLIC

        transform = T.Compose([T.ToTensor(), ToSLIC(n_segments=75)])
        dataset = MNIST('/tmp/MNIST', download=True, transform=transform)

    Args:
        add_seg (bool, optional): If set to `True`, will add the segmentation
            result to the data object. (default: :obj:`False`)
        add_img (bool, optional): If set to `True`, will add the input image
            to the data object. (default: :obj:`False`)
        **kwargs (optional): Arguments to adjust the output of the SLIC
            algorithm. See the `SLIC documentation
            <https://scikit-image.org/docs/dev/api/skimage.segmentation.html
            #skimage.segmentation.slic>`_ for an overview.
    """
    def __init__(self, add_seg=False, add_img=False, **kwargs):
        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def __call__(self, img, n_seg=40,n=200):
        # print("input image shape",img.shape)
        #reshape to ch last
        img = img.permute(1, 2, 0)
        segments_slic = segmentation.slic(img.numpy(),
                                          n_segments=n_seg,
                                          compactness=10,
                                          sigma=1,
                                          start_label=0)

        seg = torch.from_numpy(segments_slic)
        # print("seg",seg.shape)
        # rag = future.graph.rag_mean_color(img[:, :, :],
        #                     segments_slic,
        #                     connectivity=2,
        #                     mode='similarity',
        #                     sigma=255.0,
        #                    )
        # print("rag",rag.shape)
        # img = torch.from_numpy(img)

        
        h, w, c = img.shape
        # pinta ll shapes
        #   print(seg.shape,img.shape,mask.shape)
        x = scatter_mean(img.view(h * w, c), seg.view(h * w), dim=0)

        pos_y = torch.arange(h, dtype=torch.float)
        pos_y = pos_y.view(-1, 1).repeat(1, w).view(h * w)
        pos_x = torch.arange(w, dtype=torch.float)
        pos_x = pos_x.view(1, -1).repeat(h, 1).view(h * w)

        pos = torch.stack([pos_x, pos_y], dim=-1)
        pos = scatter_mean(pos, seg.view(h * w), dim=0)

        # edge_index = np.asarray([[n1, n2] for (n1, n2) in rag.edges
                                #  ]).reshape(2, -1)  #connectivity coodinates
        # weights = np.asarray([w[2]['weight'] for w in rag.edges.data()])
        # x = np.asarray([n[1]['mean color'] for n in rag.nodes.items()])

        #   lc = future.graph.show_rag(seg, rag, img[:,:,:3])

        #   pos= np.asarray([n[1]['centroid'] for n in rag.nodes.items()])
        m = int(sqrt(len(x)))

        data = Data(x=torch.tensor(x[:n]).float(),
                    pos=pos,
                edge_index=grid(m,m)[0]
                    # edge_weight=torch.tensor(weights).unsqueeze(1),
        )

        return data
class medmnist_modified(MedMNIST2D):

    def __init__(self, root, transform=None, download=None,split=None,flag="",num_classes=9):
        transfrom = T.Compose([T.Resize((64, 64)), T.ToTensor(), ImgToGraph()])
        self.flag=flag
        super().__init__(root=root, transform=transfrom,split=split,download=download)
        self.num_classes = num_classes  

    def __getitem__(self, index):
        data,y = super().__getitem__(index)
        data.y = torch.tensor(y)
        return data
class medmnist_modified_spltis(MedMNIST2D):

    def __init__(self,train=None,val=None,test=None, transform=None):
        # super().__init__(split, transform, target_transform, download, as_rgb, root)
        
        self.num_classes = train.num_classes
        self.train = train
        self.val = val 
        self.test = test
        #add keys to data


    def __getitem__(self, index):
        # just get from train
        return self.train[index]


def get_embedding_vae(x,vae):

	x_encoded = vae.encoder(x)
	mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
	std = torch.exp(log_var / 2)
	q = torch.distributions.Normal(mu, std)
	z = q.rsample()
	return z
def populateS(labels,n_clusters=8,s=None):
    """"
    Calculates the S cluster assigment transform of input patch features 
    and returns the (S) and the aggregated (out_adj) as well.
    shape : ( number of patches , number of clusters)
    """
    # print("S is " ,s==None)
    n_patches=len(labels)
    div = int(sqrt(n_patches))
    if s == None:
        s = np.zeros((n_patches,n_clusters))
        for i in range(s.shape[0]):
            s[i][labels[i]] = 1
         # TODO optimise this!
    else:
        s=s

    #calc adj matrix
    adj = to_dense_adj(grid(n_patches//div,n_patches//div)[0]).reshape(n_patches,n_patches)
    return s , np.matmul(np.matmul(s.transpose(1, 0),adj ), s)

from torch_geometric.data import Data,Dataset
from torch_geometric.utils import to_dense_adj, grid,dense_to_sparse

from monai.data import GridPatchDataset, DataLoader, PatchIter
def filter_a(data):

    if data.y==3:
        return False
    else:
        True

class ImageTOGraphDataset(Dataset):
    """ 
    Dataset takes holds the kmaens classifier and vae encoder. On each input image we encode then get k mean label then formulate graph as Data object
    """
    def __init__(self,data,vae,kmeans,norm_adj=True,return_x_only=None):
        self.data=data
        self.vae=vae
        self.return_x_only=return_x_only
        with open(kmeans,"rb") as f:
            self.kmeans = pickle.load(f)      
        self.norm_adj = norm_adj
        self.patch_iter = PatchIter(patch_size=(32, 32), start_pos=(0, 0))
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        
        patches = []
        for x in self.patch_iter(self.data[index][0]):
            patches.append(x[0])
            
        patches = torch.stack([torch.tensor(np.array(patches))],0).squeeze()

        z=get_embedding_vae(patches,self.vae).clone().detach().cpu().numpy()
        label=self.kmeans.predict(z)
        s,out_adj = populateS(label)
        x = np.matmul(s.transpose(1,0) , z)
        if self.return_x_only:
            return x,label,self.data[index][1]
        #if normlaise adj 
        if self.norm_adj:
            out_adj = out_adj.div(out_adj.sum(1))
            #nan to 0 in tensor 
            out_adj = out_adj.nan_to_num(0)
            #assert if there is nan in tensor
            assert out_adj.isnan().any() == False , "Found nan in out_adj"
        return Data(x=torch.tensor(x).float(),edge_index=dense_to_sparse(out_adj)[0],y=torch.tensor(self.data[index][1]),edge_attr=dense_to_sparse(out_adj)[1])

class KDataset(Dataset):
    
    """
    Dataset to store the cluster repr of images as embedding feature maps 
    and edge as a grid formation ? - not really we use cluster feature relations as a 

    """

    def __init__(self,zs,data_128,labels,root=None,transform=None,pre_transform=None,pre_filter=None):
        super(KDataset,self).__init__(root,transform,pre_transform,pre_filter)
        self.zs=zs
        self.data = data_128
        self.labels = labels
    

    def __getitem__(self,index):
        
        s,out_adj = populateS(self.labels[index])
        x = np.matmul(s.transpose(1,0) , self.zs[index])

        
        return Data(x=x,edge_index=dense_to_sparse(out_adj)[0],y=torch.tensor(self.data[index][1]),edge_attr=dense_to_sparse(out_adj)[1])
    def __len__(self):
        return len(self.zs)


from torch_geometric.loader import DataLoader as GraphDataLoader

# ImData = ImageTOGraphDataset(data=data_128,vae=vae,kmeans=k)
#print dataset stats
from torch import nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import pytorch_lightning as pl

class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(latent_dim=latent_dim,
                                        input_height=input_height,
                                        first_conv=False,
                                        maxpool1=False)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu),
                                       torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        # print(batch)
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss_ = self.gaussian_likelihood(x_hat, self.log_scale, x) # old recon_loss 
        # print(recon_loss.shape)
        recon_loss = torch.nn.MSELoss()(x_hat,x)
        # print(recon_loss.shape)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss_) # with old recon_loss
        # elbo = (kl + recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss_': recon_loss.mean(),
            'recon_loss': recon_loss_.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo



class ImageToClusterHD5(Dataset):
    """ 
    Dataset takes holds the kmaens classifier and vae encoder. On each input image we encode then get k mean label then formulate graph as Data object
    """
    def __init__(self,data,norm_adj=True,split=None,n_clusters=None):
        #read h5 file into self .data
        self.data = h5py.File(data,'r')
        self.x = self.data['x']
        self.ys = self.data['ys'][:].reshape(-1)
        self.labels = self.data['edge_index'][:].reshape(-1,self.data['edge_index'].shape[2])
        self.nc = n_clusters
        print("number of clusters", self.nc)
        print("len of x", self.x.shape)
        print("len of l", self.labels.shape)
        print("len of y", self.ys.shape)
        self.x= self.x[:self.labels.shape[0]]
        self.ys = self.ys[:self.labels.shape[0]]
        print("->len of x", self.x.shape)
        print("->len of l", self.labels.shape)
        print("->len of y", self.ys.shape)

        assert len(self.x) == len(self.labels) , "x and labels should be same length"
        self.norm_adj = norm_adj
        # self.num_classes=9 They have a builtin property for this
        if split == 'train':
            self.x = self.x[:int(len(self.x)*.8)]
            self.labels = self.labels[:int(len(self.labels)*.8)]
            self.ys = self.ys[:int(len(self.ys)*.8)]
        elif split == 'val':
            self.x = self.x[int(len(self.x)*.8):]
            self.labels = self.labels[int(len(self.labels)*.8):]
            self.ys = self.ys[int(len(self.ys)*.8):]        
        
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):

        label = self.labels[index]
        s,out_adj = populateS(label,n_clusters=label.shape[0] if self.nc == None else self.nc)
        x = self.x[index][:]
        if self.norm_adj:
            out_adj = out_adj.div(out_adj.sum(1))
            #nan to 0 in tensor 
            out_adj = out_adj.nan_to_num(0)
            #assert if there is nan in tensor
            assert out_adj.isnan().any() == False , "Found nan in out_adj"
        # assert self.ys[index] != None , "Found None in ys"
        return Data(x=torch.tensor(x).float(),edge_index=dense_to_sparse(out_adj)[0],y=torch.tensor([self.ys[index]]),edge_attr=dense_to_sparse(out_adj)[1])
