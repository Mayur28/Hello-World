import torch.utils.data

def CreateDataset(opt):
    dataset = None

    from data.unaligned_dataset import UnalignedDataset
    dataset = UnalignedDataset() #Retrieves the images from A and B and applies the necessary transformations on the images
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset #This seems to make sense!


class CustomDatasetDataLoader:
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
      self.opt=opt
      self.dataset = CreateDataset(opt)
      self.dataloader = torch.utils.data.DataLoader(self.dataset,batch_size=opt.batchSize,shuffle=not opt.serial_batches, num_workers=int(opt.nThreads))# This is a loader fror images in A and B. THIS IS ONE OF THE FOUNDATIONS OF THE ENTIRE ALGORITHM. By the time that we are calling self.dataloader, the entire dataset is already loaded, we are just extracted the batches on a need to know basis.

    def load_data(self):
        return self.dataloader

    def __len__(self):# We do use this function in train.py... It seems that the __ is when we are overwriting functions
        return min(len(self.dataset), self.opt.max_dataset_size)
