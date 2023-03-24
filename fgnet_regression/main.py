"""
Summary: This script trains a residual neural decision forest (RNDF) for facial age 
estimation (on CACD dataset). It applies the simple idea of residual learning 
to neural decision forest (NDF). Residual learning is widely adopted in CNN, 
here let's use it for NDF.
Author: Nicholas Li
Contact: Nicholas.li@connect.ust.hk
License: MIT
"""

# utility functions
import data_prepare # dataset preparation
import model # model implementation
import trainer # training functions
import optimizer # optimization functions
import utils # other utilities
from data_prepare import FacialAgeDataset

# public libraries
import torch
import logging
import numpy as np
import time

def main():
    # logging configuration
    logging.basicConfig(level=logging.INFO,
        format="[%(asctime)s]: %(message)s"
    )
        
    # parse command line input
    opt = utils.parse_arg()

    device = torch.device("cpu")
        
    # prepare training and validation dataset
    vars = data_prepare.prepare_db(opt)
    
    # sanity check for FG-NET dataset, not used for now   
    # assertion: the total images in the eval set lists should be 1002
    # total_eval_imgs = sum([len(db['eval'][i]) for i in range(len(db['eval']))])
    # print(total_eval_imgs)
    # if db['train'][0].name == 'FGNET':
    #     assert total_eval_imgs == 11, 'The preparation of the evalset is incorrect.'
    
    # training     
    if opt.train:
            best_MAEs = []
            last_MAEs = []
            # record the current time
            opt.save_dir += time.asctime(time.localtime(time.time()))
            # for FG-NET, do training multiple times for leave-one-out validation

            keys = list(vars.keys())
            mae_score = []
            for i in range(len(keys)):
            # Get the training data and the testing data using the current key
                test_key = keys[i]
                train_set = {}
                eval_set = {}
                for key in keys:
                    if key != test_key:
                        train_set[key] = vars[key]
                eval_set[test_key] = vars[test_key]
                
                train_list = []
                train_list.append(FacialAgeDataset(train_set, opt, 'train'))
                
                eval_list = []
                eval_list.append(FacialAgeDataset(eval_set, opt, 'eval'))
                
                db = {'train':train_list, 'eval':eval_list} 


            # for CACD, do training just once
                for exp_id in range(len(db['train'])):
                    # initialize the model
                    model_train = model.prepare_model(opt)
                    
                    # configurate the optimizer and learning rate scheduler
                    optim, sche = optimizer.prepare_optim(model_train, opt)
                    
                    # train the model and record mean average error (MAE)
                    model_train, MAE, last_MAE = trainer.train(model_train, optim, sche, db, opt, exp_id)
                    best_MAEs += MAE
                    last_MAEs.append(last_MAE.data.item())


                    # remove the trained model for leave-one-out validation
                    if exp_id != len(db['train']) - 1:
                        del model_train


                avg_best_maes = np.mean(best_MAEs)
                mae_score.append(avg_best_maes)
                
                np.save('./MAE.npy', np.array(best_MAEs))
                np.save('./Last_MAE.npy', np.array(last_MAEs))

                # save the final trained model
                utils.save_model(model_train, opt)
                torch.save(model_train,'./FGNET_model.pth')

            avg_mae_score = np.mean(mae_score)
            print(avg_mae_score,"Average mae score")

            
            torch.save(model_train,'./FGNET_main_model.pth')

        # testing a pre-trained model    
    elif opt.evaluate:
            # path to the pre-trained model
            save_dir = opt.test_model_path
            # example: save_dir = '../model/CACD_MAE_4.59.pth'
            example: save_dir = './FGNET_model.pth'
            model_loaded = torch.load(save_dir)        
            # test the model on the evaluation set 
            # the last subject is the test set (compatible with FG-NET)
            trainer.evaluate(model_loaded, db['eval'][-1], opt)    
       
    return 

if __name__ == '__main__':
    main()