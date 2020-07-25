import torch.utils.data

def CreateDataset(opt):
    dataset = None

    from data.unaligned_dataset import UnalignedDataset
    dataset = UnalignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset #This seems to make sense!


class CustomDatasetDataLoader:
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
		self.opt=opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,batch_size=opt.batchSize,shuffle=not opt.serial_batches, num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
