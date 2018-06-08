''' This is dataset loader file to load the bAbI dataset. '''
import re
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate

class adict(dict):
	def __init__(self, *args, **kargs):
		dict.__init__(self, *args, **kargs)
		self.__dict__ = self

def pad_collate(batch):
	max_len_ques = float('-inf')
	max_sen_len_context = float('-inf')
	max_len_context = float('-inf')
	
	for item in batch:
		contexts, ques, _ = item
		if len(contexts) > max_len_context:
			max_len_context = len(contexts)
		if len(ques) > max_len_ques:
			max_len_ques = len(ques)
		for sen in contexts:
			if(len(sen) > max_sen_len_context):
				max_sen_len_context = len(sen)
	max_len_context = min(max_len_context, 70)
	for idx, item in enumerate(batch): # Going through each example in the batch which contains their ow context, question and answer.
		context_i, question, answer = item
		context_i = context[-max_len_context:] #???
		context = np.zeros((max_len_context, max_sen_len_context))
		for i, sen in enumerate(context_i): # going through ith context containing max_len_context sentences and a question
			context[i] = np.pad(sen, (0, max_sen_len_context-len(sen)), 'constant', constant_values=0)
		question = np.pad(question, (0, max_len_ques-len(question)), 'constant', constant_values=0)
		batch[idx] = (context, question, answer)
		
	return default_collate(batch)

class BabiDataSet(Dataset):
	def __init__(self, task_id, mode='train'):
		self.mode = mode
		self.vocab_path = 'dataset/babi{}_vocab.pkl'.format(task_id)
		train_data, test_data = get_train_test(task_id) # Get raw train_data and test_data from babi dataset
		self.QA = adict()
		self.QA.VOCAB = {'<PAD>': 0, '<EOS>':1}
		self.QA.INV_VOCAB = {0:'<PAD>', 1:'<EOS>'}
		self.train = self.get_processed_data(train_data)
		self.val = [self.train[i][int(9*len(self.train[i])/10):] for i in range(3)] # splitting into 90/10 train/val dataset
		self.train = [self.train[i][:int(9*len(self.train[i])/10)] for i in range(3)] # splitting into 90/10 train/val dataset
		self.test = self.get_processed_data(test_data)
	
	def set_mode(self, mode):
		self.mode = mode #????
		
	def __len__(self):
		if self.mode == 'train':
			return len(self.train[0])
		elif self.mode == 'val':
			return len(self.val[0])
		elif self.mode == 'test':
			return len(self.test[0])
		else:
			print ("Invalid Mode!")
			return
	
	def __getdata__(self, index):
		if self.mode == 'train':
			contexts, questions, answers = self.train
		elif self.mode == 'val':
			contexts, questions, answers = self.val
		elif self.mode == 'test':
			contexts, questions, answers = self.test
		
		return contexts[index], questions[index], answers[index]
	
	def get_processed_data(self, raw_data):
	    unindexed= get_unprocessed_data(raw_data)
	    questions=[]
	    contexts= []
	    answers= []
	    for qa in unindexed:
	        context= [c.lower().split()+ ['<EOS>'] for c in qa['C']]

	        for con in context:
	            for token in con:
	                self.build_vocab(token)
	        context= [[self.QA.VOCAB[token] for token in sentence] for sentence in context]
	        question= qa['Q'].lower().split()+ ['<EOS>']

	        for token in question:
	            self.build_vocab(token)
	        question= [self.QA.VOCAB[token] for token in question]
	        
	        self.build_vocab(qa['A'].lower())
	        answer= self.QA.VOCAB[qa['A'].lower()]

	        contexts.append(context)
	        questions.append(question)
	        answers.append(answer)
	        return (contexts, questions, answers)

	def build_vocab(self, token):
		if not token in self.QA.VOCAB:
	        next_index= len(self.QA.VOCAB)
	        self.QA.VOCAB[token]= next_index
	        self.QA.IVOCAB[next_index]= token
	
	
	
def get_train_test(task_id):
	paths = glob('data/en-10k/qa{}_*'.format(task_id))
	for path in paths:
		if 'train' in path;
			with open(path, 'r') as f:
				train = f.read()
		elif 'test' in path:
			with open(path, 'r') as f:
				test = f.read()
	
	return train, test
	

def build_vocab(raw_data):
	lowered= raw_data.lower()
	tokens= re.findall('[a-zA-Z]+',lowered)
	types= set(tokens)
	return types


def get_unprocessed_data(raw_data):
    tasks = []
    task = None
    data = raw_data.strip().split('\n')
    for i, line in enumerate(data):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "S": ""}
            counter = 0
            id_map = {}

        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        # if not a question
        if line.find('?') == -1:
            task["C"] += line + '<line>'
            id_map[id] = counter
            counter += 1
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = [] # Supporting facts
            for num in tmp[2].split():
                task["S"].append(id_map[int(num.strip())])
            tc = task.copy()
            tc['C'] = tc['C'].split('<line>')[:-1]
            tasks.append(tc)
    return tasks





if __name__ == '__main__':
    dataset_train= BabiDataset(20, mode='train') # Loading the dataset with task_id = 20
    train_loader= DataLoader(dataset_train,batch_size=2, shuffle=True,collate_fn= pad_collate)
    for batch_idx, data in enumerate(train_loader):
        contexts, questions, answers= data
        break		
