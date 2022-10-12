import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    WIPTest=[]
    one=0
    two=0
    three=0
    four=0
    for i, data in enumerate(dataset):
        if opt.no_input==2:
            if(data['A1_paths']==['./datasets/MRI2CT_inOutJoin/trainA_in/5IMG-0016-00052.png'] and one==0):
                one=1
                print('found')
                WIPTest.append(data)
            if(data['A1_paths']==['./datasets/MRI2CT_inOutJoin/trainA_in/1IMG-0004-00032.png'] and two==0):
                two=1
                WIPTest.append(data)
            if(data['A1_paths']==['./datasets/MRI2CT_inOutJoin/trainA_in/22IMG-0005-00044.png'] and three==0):
                three=1
                WIPTest.append(data)
            if(data['A1_paths']==['./datasets/MRI2CT_inOutJoin/trainA_in/34IMG-0005-00060.png'] and four==0):
                four=1
                WIPTest.append(data)
        else:
            if(data['A1_paths']==['./datasets/MRI2CT_inOutJoin/trainA_in/5IMG-0016-00052.png'] and one==0):
                one=1
                print('found')
                WIPTest.append(data)
            if(data['A1_paths']==['./datasets/MRI2CT_inOutJoin/trainA_inT2/1IMG-0004-00032.png'] and two==0):
                two=1
                WIPTest.append(data)
            if(data['A1_paths']==['./datasets/MRI2CT_inOutJoin/trainA_inT2/22IMG-0005-00044.png'] and three==0):
                three=1
                WIPTest.append(data)
            if(data['A1_paths']==['./datasets/MRI2CT_inOutJoin/trainA_inT2/34IMG-0005-00060.png'] and four==0):
                four=1
                WIPTest.append(data)
        #print(str(data['A1_paths']) + ' ' + str(data['A2_paths']))
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        #print(data)
        model.optimize_parameters()

        #if total_steps % opt.display_freq == 0:
         #   visualizer.display_current_results(model.get_current_visuals(), epoch)

        #if total_steps % opt.print_freq == 0:
            #errors = model.get_current_errors()
            #t = (time.time() - iter_start_time) / opt.batchSize
            #visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            #if opt.display_id > 0:
             #   visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        #if total_steps % opt.save_latest_freq == 0:
         #   print('saving the latest model (epoch %d, total_steps %d)' %
           #       (epoch, total_steps))
            #model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
    index=1
    for data in WIPTest:
        model.set_input(data)
        model.test()
        visualizer.display_current_results(model.get_current_visuals(), epoch, index)
        index=index+1
        
    
