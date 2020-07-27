import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from util.visualizer import Visualizer
from models.single_model import SingleModel

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)

opt = TrainOptions().parse()# Sort of tokenize all the options
config = get_config(opt.config)# Calls GTA --> This is a special file from nvidia!
data_loader = CreateDataLoader(opt)# The opt isnt really needed by we just need an instance of the class
dataset = data_loader.load_data()# This is an implemented function!
dataset_size = len(data_loader)#
print('#training images = %d' % dataset_size)
#All above is thoroughly understood

model = SingleModel()
model.initialize(opt)

visualizer = Visualizer(opt)

total_steps = 0
# Below is the big deal!!! range(1,100+100+1)# the lr decays for last 100 epochs
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):# This represents the chunked dataset! it will process each batch accordingly. GUARANTEED!--> Apparently restarts the dataloader iterator on each epoch. I DEF. NEED THIS!
		# Each pass, the dataset dataloader is called which takes a slice of what is retrieved from Unaligned_Dataset, noting that recieving from Unaligned_dataset in each pass is in the desired dictionary format! This means that data is in mini-dictionary form!
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data) #Remember at this stage, data is the batch 'dataset' in dictionary format. It slots the data into the correct variables self.inputA,etc to easily perform propagation operations
        model.optimize_parameters(epoch) # This is understood (the idea), still need to go deeper (into the actual functions that make it up)

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors(epoch)
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            #if opt.display_id > 0:
            #    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if opt.new_lr:# We're not updating the learning like this!
        if epoch == opt.niter:
            model.update_learning_rate()
        elif epoch == (opt.niter + 20):
            model.update_learning_rate()
        elif epoch == (opt.niter + 70):
            model.update_learning_rate()
        elif epoch == (opt.niter + 90):
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
    else:
        if epoch > opt.niter:
            model.update_learning_rate()
