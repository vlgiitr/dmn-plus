import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as Variable
import torch.utils.data as DataLoader

class QuestionModule(nn.Module):
	def __init__(self, vocab_size, hidden_size):
		super(QuestionModule, self).__init__()
		self.vocab_size = vocab_size # Size of the vocabulary used in word embedding
		self.hidden_size = hidden_size # Size of the hidden state of GRU
		self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

	def forward(self, questions, word_embedding):
		# questions.size() = (batch_size, num_tokens)
		# word_embedding -> (batch_size, num_tokens, embedding_length)
		# self.gru() -> (1, batch_size, hidden_size)

		questions = word_embedding(questions) # Word embedding of the question
		output, questions = self.gru(questions) # What is the initial hidden vector given to GRU?
		questions = torch.transpose(questions, 0, 1)

		return questions

class InputModule(nn.Module):
	def __init__(self, vocab_size, hidden_size):
		super(InputModule, self).__init__()
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
		for name, param in self.gru.state_dict().items():
			if 'weight' in name:
				init.xavier_normal(param)
		self.dropout = nn.Dropout(0.1)

	''' We will now define the encoding scheme which is positional encoding in the paper " Dynamic Memory Network for Textual and Visual 
    Question Answering '''
	def positional_encoder(embedded_sentence):
		# embedded_sentence.size() = (batch_size, num_sentences, num_tokens, embedding_length)
		# l.size() = (num_tokems, embedding_length)
		# output.size() = (num_batch, num_sentences, embedding_length)
		# The outputs are basically f1, f2, f3,.... which will go into the input fusion layer in the next step to add share information
		# between sentences using a BiDirfectional GRU module.

		batch_size, num_sentences, num_tokens, embedding_length = embedded_sentence.size()
		l = [] # It will be same for all sentences in all batches as num_tokens and embedding_length is same for the entire dataset.
		for j in range(num_tokens):
			x = []
			for d in range(embedding_length):
				x.append((1 - (j/(num_tokens-1))) - (d/(embedding_length-1)) * (1 - 2*j/(num_tokens-1)))
			l.append(x)
		
		l = torch.FloatTensor(l)
		l = l.unsqueeze(0) # adding an extra dimension at first place for batch_size
		l = l.unsqueeze(1) # adding an extra dimension at sencond place for num_sentences
		l = l.expand_as(embedded_sentence) # so that l.size() = (batch_size, num_sentences, num_tokens, embedding_length)

		mat = embedded_sentence*Variable(l.cuda())
		f_ids = torch.sum(mat, dim=2).squeeze(2) # sum along token dimension

		return f_ids


	def forward(self, input, word_embedding):
		# input.size() = (batch_size, num_sentences, num_tokens)
		# word_embedding -> (batch_size, num_sentences, num_tokens, embedding_length)
		# positional_encoder(word_embedding(input)) -> (batch_size, num_sentences, embedding_length)
		# Now BidirectionalGRU blocks receive their input, the output of the positional encoder and finally give facts
		# facts.size() = (batch_size, num_sentences, embedding_length) embedding_length = hidden_size

		input = input.view(input.size()[0], -1)# Isn't it already in this format ?
		input = word_embedding(input)
		input = input.view(input.size()[0], input.size()[1], input.size()[2], -1)
		input = self.positional_encoder(input)
		input = self.dropout(input)

		h0 = Variable(torch.zeros(2, input.size()[0], self.hidden_size).cuda()) # Initializing the initial hidden state (at t=0 time step)
		facts, hdn = self.gru(input, h0)
		facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]

		return facts

class AttnGRUCell(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(AttnGRUCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.Wr = nn.Linear(input_size, hidden_size)
		self.Ur = nn.Linear(hidden_size, hidden_size)
		self.W = nn.Linear(input_size, hidden_size)
		self.U = nn.Linear(hidden_size, hidden_size)

		init.xavier_normal(self.Wr.state_dict()['weight'])
		init.xavier_normal(self.Ur.state_dict()['weight'])
		init.xavier_normal(self.W.state_dict()['weight'])
		init.xavier_normal(self.U.state_dict()['weight'])

	def forward(self, fact, hi_1, g):
		# fact is the final output of InputModule for each sentence and fact.size() = (batch_size, embedding_length)
		# hi_1.size() = (batch_size, embedding_length=hidden_size)
		# g.size() = (batch_size, )

		r_i = F.sigmoid(self.Wr(fact) + self.Ur(hi_1))
		h_tilda = F.tanh(self.W(fact) + r*self.U(hi_1))
		hi = g*h_tilda + (1 - g)*hi_1

		return hi # Returning the next hidden state considering the first fact and so on.

class AttnGRU(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(AttnGRU, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.AttnGRUCell = AttnGRUCell(input_size, hidden_size)

	def forward(self, facts, G):
		# facts.size() = (batch_size, num_sentences, embedding_length)
		# fact.size() = (batch_size, embedding_length=hidden_size)
		# G.size() = (batch_size, num_sentences)
		# g.size() = (batch_size, )

		h_0 = Variable(torch.zeros(self.hidden_size)).cuda()

		for sen in range(facts.size()[1]):
			fact = facts[:, sen, :]
			g = G[:, sen]
			if sen == 0: # Initialization for first sentence only 
				hi_1 = h_0.unsqueeze(0).expand_as(fact)
			hi_1 = self.AttnGRUCell(fact, hi_1, g)
		C = hi_1 # Final hidden vector as the contextual vector used for updating memory

		return C

class MemoryModule(nn.Module): # Takes facts, question and prev_mem as its and output next_mem
	def __init__(self, hidden_size):
		super(MemoryModule, self).__init__()
		self.hidden_size = hidden_size
		self.AttnGRU = AttnGRU(hidden_size, hidden_size)
		self.W1 = nn.Linear(4*hidden_size, hidden_size)
		self.W2 = nn.Linear(hidden_size, 1)
		self.W_mem = nn.Linear(3*hidden_size, hidden_size)

		init.xavier_normal(self.W1.state_dict()['weight'])
		init.xavier_normal(self.W2.state_dict()['weight'])
		init.xavier_normal(self.W_mem.state_dict()['weight'])

	def gateMatrix(self, facts, questions, prev_mem):
		# facts.size() = (batch_size, num_sentences, embedding_length=hidden_size)
		# questions.size() = (batch_size, 1, embedding_length)
		# prev_mem.size() = (batch_size, 1, embedding_length)
		# z.size() = (batch_size, num_sentences, 4*embedding_length)
		# G.size() = (batch_size, num_sentences)

		questions = questions.expand_as(facts)
		prev_mem = prev_mem.expand_as(facts)

		z = torch.cat([facts*questions, facts*prev_mem, torch.abs(facts - questions), torch.abs(facts - prev_mem)], dim=2)
		# z.size() = (batch_size, num_sentences, 4*embedding_length)
		z = z.view(-1, 4*embedding_length)
		Z = self.W2(F.tanh(self.W1(z)))
		Z = Z.view(batch_size, -1)
		G = F.softmax(Z)

		return G

	def forward(self, facts, questions, prev_mem):
		# questions = questions.unsqueeze(1)
		# prev_mem = prev_mem.unsqueeze(1)
		G = self.gateMatrix(facts, questions, prev_mem)
		C = self.AttnGRU(facts, G)
		# Now considering prev_mem, C and question, we will update the memory state as follows
		concat = torch.cat([prev_mem.squeeze(1), C, questions.squeeze(1)], dim=1)
		next_mem = F.relu(self.W_mem(concat))
		next_mem = next_mem.unsqueeze(1)

		return next_mem

class AnswerModule(nn.Module):
	def __init__(self, vocab_size, hidden_size):
		super(AnswerModule, self).__init__()
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.W = nn.Linear(2*hidden_size, vocab_size)
		init.xavier_normal(self.W.state_dict()['weight'])
		self.dropout = nn.Dropout(0.1)

	def forward(self, final_mem, questions):
		final_mem = self.dropout(final_mem)
		concat = torch.cat([final_mem, questions], dim=2).squeeze(1)
		out = self.W(concat) # As per the paper, we are concatenating the final memory state m_T, and the question q and passing 
		# this resultant vector to a linear layer

		return out


''' We define the model for the network incorporating the input, question, answer and the episodic memory module. We use the Cross Entropy loss criterion for measuring loss'''    
class DMN(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_pass=3, qa=None):
        super(DMN,self).__init__()
        self.num_pass= num_pass
        self.qa= qa
        self.word_embedding= nn.Embedding(vocab_size, hidden_size, padding_index=0, sparse=True)
        init.uniform(self.word_embedding.state_dict()['weight'], a= -(3**0.5), b=3**0.5)
        self.criterion= nn.CrossEntropyLoss(size_average=False)
        
        self.input_module= input_module(vocab_size,hidden_size)   ##Vocab size refers to the size of vocabulary used
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
        pred= F.softmax(output)
        _, pred_id= torch.max(pred, dim=1)
        correct= (pred_id.data == answers.data)
        acc= torch.mean(correct.float())   
        return loss+para_loss, acc
    
    def interpret_indexed_tensor(self,var):
        if len(var.size()) == 3:
            for n, sentences in enumerate(var):
                s= ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                print (str(n)+'th batch, '+str(i)+'th sentence, '+str(s))
                
        elif len(var.size()) == 2:
            for n, sentence in enumerate(var):
                s= ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                print (str(n)+'th batch, '+str(s))
                
        elif len(var.size()) == 1:
            for n, token in enumerate(var):
                s= self.qa.IVOCAB[token.data[0]]
                print (str(n)+'th of batch, '+str(s))
        
