import torch
import torch.nn as nn
import torch.nn.functional as f
import os
import torch.nn.init as init
import torch.autograd as Variable
import torch.utils.data as DataLoader
import numpy as np

''' Dynamic Memory Networks for Visual and Textual Question Answering. We define the model for the network incorporating 
    the input, question, answer and the episodic memory module. We use the Cross Entropy loss criterion for measuring loss'''    
class DMN(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_pass=3, qa=None):
        super(DMN,self).__init__()
        self.num_pass= num_pass
        self.qa= qa
        self.word_embedding= nn.Embedding(vocab_size, hidden_size, padding_index=0, sparse=True)
        init.uniform(self.word_embedding.state_dict()['weight'], a= -(3**0.5), b=3**0.5)
        self.criterion= nn.CrossEntropyLoss(size_average=False)
        
        self.input_module= input_module(vocab_size,hidden_size)   ##Vocab size refers to the size of vocabulary used###
        self.question_module= question_module(vocab_size, hidden_size) 
        self.memory= episodic_memory(hidden_size)
        self.answer_module= answer_module(vocab_size,hidden_size)
        
    def forward(self, context, questions):
        #facts.size()= (batch_size, num_sentences, embedding_length= hidden.size()) 
        #questions.size() = (batch_size, 1, embedding_length)
        facts= self.input_module(context, self.word_embedding)
        questions= self.question_module(questions, self.word_embedding)
        X= questions
        for passes in range(self.num_pass):
            X= self.memory(facts, questions, X)
        pred= self.answer_module(X, questions)
        return pred_id
    
    '''Total loss to be calculated '''
    
    def loss(self,context, questions, targets):
        output= self.forward(context, questions)
        loss= self.criterion(output, targets)
        para_loss= 0
        for param in self.parameters():
            para_loss+= 0.001* torch.sum(param*param)
        pred= f.softmax(output)
        _, pred_id= torch.max(pred, dim=1)
        correct= (pred_id.data == answers.data)
        acc= torch.mean(correct.float())   
        return loss+reg_loss, acc
    
    def interpret_indexed_tensor(self,var):
        if len(var.size()) == 3:
            for n, sentences in enumerate(var):
                s= ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                print '{n}th batch, {i}th sentence, {s}'
                
        elif len(var.size()) == 2:
            for n, sentence in enumerate(var):
                s= ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                print '{n}th batch, {s}'
                
        elif len(var.size()) == 1:
            for n, token in enumerate(var):
                s= self.qa.IVOCAB[token.data[0]]
                print '{n}th of batch, {s}'
        