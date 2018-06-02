''' This file contains the code for training and testing the model. Adam optimizer is used for training with a 
learning rate of 0.001 and a batch size of 128. Training is done for 256 epochs with early stopping 
if validation loss doesn't decrease within last 20 epochs. Weights are initialized  using Xavier Initialization 
except for word embeddings. Dropout and L2 are used as regularization methos on sentence encodings and answer module.'''


import torch
import torch.nn as nn
import torch.nn.functional as f
import os
import torch.nn.init as init
import torch.autograd as Variable
import torch.utils.data as DataLoader
import numpy as np




if __name__ == '__main__':
    for itr in range(10):
        for task in range(1,21):
            dataset= BabiDataset(task)
            vocab_size= len(dataset.QA.VOCAB)
            hidden_size= 80
            
            model= DMN(hidden_size, vocab_size, num_pass= 3, qa= dataset.QA)   ##vocab_size denotes the size of word embedding used 
            early_stop_count= 0
            early_stop_flag= False
            best_acc= 0
            optim= torch.optim.Adam(model.parameters())
            
            for epoch in range(256):
                dataset.set_mode('train')
                train_load= DataLoader(dataset, batch_size=100, shuffle= True, collate_fn= pad_collate)  ### Loading the babi dataset
                
                model.train()                                                       ### training the network
                if not early_stop_flag:
                    total_acc=0
                    count= 0
                    for batch_id, data in enumerate(train_load):
                        optim.zero_grad()
                        context, questions, answers = data
                        batch_size= context.size()[0]
                        context= Variable(context.long())                           ## context.size() = (batch_size, num_sentences, embedding_length) embedding_length = hidden_size 
                        questions= Variable(questions.long())                       ## questions.size() = (batch_size, num_tokens)
                        answers= Variable(answers)
                        
                        total_loss, acc= model.loss(context,questions,answers)      ## Loss is calculated and gradients are backpropagated through the layers.
                        total_loss.backward()
                        total_acc+= acc*batch_size
                        count+= batch_size
                        
                        if batch_id %20 == 0:
                            print('training error')
                            print ('task '+str(task_id)+',epoch '+str(epoch)+',loss ' +str(loss.data[0])+',total accuracy : '+str(total_acc/cnt))
                        optim.step()
                    
                    '''Validation part'''


                    dataset.set_mode('valid')
                    valid_load = DataLoader(dataset, batch_size= 100, shuffle= False, collate_fn= pad_collate)    ## Loading the validation data
                    
                    model.eval()
                    total_acc=0
                    count=0
                    for batch_id, data in enumerate(train_load):
                        context, questions, answers = data
                        batch_size= context.size()[0]
                        context= Variable(context.long())
                        questions= Variable(questions.long())
                        answers= Variable(answers)
                        
                        _, acc= model.loss(context,questions,answers)  
                        total_loss.backward()
                        total_acc+= acc*batch_size
                        count+= batch_size
                    
                    total_acc= total_acc/ count
                    
                    if total_acc > best_acc:
                        best_acc= total_acc
                        best_state= model.state_dict()
                        early_stop_count= 0
                    else:
                        early_stop_count+= 1                   
                        if early_stop_count > 20:
                            early_Stop_flag= True
                    
                    print ('itr '+str(itr)+',task_id '+str(task_id)+',epoch '+str(epoch)+',total_acc '+str(total_acc))
                    
                    with open('log.txt', 'a') as fp:
                        fp.write('itr '+str(itr)+', task_id '+str(task_id)+',epoch '+str(epoch)+',total_acc '+str(total_acc)+'+\n ')
                    if total_acc == 1.0:
                        break
                else:
                    print('iteration'+str(itr)+'task' +str(task_id)+' Early Stopping at Epoch'  +str(epoch)+'validation accuracy :' +str(best_acc))
                    
            
            
            dataset.set_mode('test')
            test_load= DataLoader(dataset, batch_size=100, shuffle= False, collate_fn= pad_collate)
            
            test_acc= 0
            count=0
            
            for batch_id, data in enumerate(test_load):
                    context, questions, answers = data
                    batch_size= context.size()[0]
                    context= Variable(context.long())
                    questions= Variable(questions.long())
                    answers= Variable(answers)
                    
                    model.load_state_dict(best_state)
                    _, acc= model.loss(context, questions, answers)
                    
                    test_acc += acc* batch_size 
                    count += batch_size
                    print ('itr '+ str(itr)+'task =' +str(task_id)+ 'Epoch ' +str(epoch)+' test accuracy : '+str(test_acc / count))
                    
                    
                    
                    os.makedirs('models',exist_ok= True)
                    with open('models/task'+str(task_id)+'_epoch'+str(epoch)+'_run'+str(run)+'_acc'+str(test_acc/cnt)+'.pth', 'wb') as fp:
                        torch.save(model.state_dict(), fp)
                    with open('log.txt', 'a') as fp:
                        fp.write('[itr '+str(itr)+', Task '+str(task_id)+', Epoch '+str(epoch)+'] [Test] Accuracy : '+str(total_acc)+' + \n')

                        
