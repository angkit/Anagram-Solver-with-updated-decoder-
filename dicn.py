import numpy as np 
import pickle

lexicon = []

with open('ptb.txt','r') as f:
    contents = f.readlines()
	
	
    for l in contents[:len(contents)]:
	    
	    all_words=l.split()
	    lexicon += list(all_words)

lexicon = list(set(lexicon));
print(lexicon)	    

vocb_size=len(lexicon)+1
print('vocb_size', vocb_size)
pad=vocb_size

voc = lexicon
vocab=dict([(x, y) for (y, x) in enumerate(voc)])
rev_vocab=dict([(x, y) for (x, y) in enumerate(voc)])




# Input data_file

with open('ptb.txt','r') as f:
	
    contents = f.readlines()

x=[]
for sentence in contents:
    
    a=sentence;
    d=sorted(a);
    token=[vocab.get(w) for w in d]
    n_token=[]
    for i in token:
        if(i!= None):
            n_token.append(i)
    
    

    x.append(n_token)
max_len=0
ipf=[]
for i in x:
    if max_len<=len(i):
       max_len=len(i)
    else: max_len=max_len   

for i in x:
	j=[]
	j=np.lib.pad(i, (0,max_len-len(i)), 'constant', constant_values=(pad))
	
	ipf.append(j)
	
data_x = np.array(ipf).T



# Train_X
testing_size=int(data_x.shape[1]-1)
train_x= data_x[:,:testing_size]

print('train_x.shape', train_x.shape)






#Getting Seq_len

seq_length=len(train_x)
print('seq_length', seq_length)
n_sents=train_x.shape[1]
print('n_sents', n_sents)


#Test_X

test_x=data_x[:,testing_size:]

print('test_x.shape', test_x.shape)


t=test_x.flatten()
tok=[rev_vocab.get(i) for i in t]
     
n_token=[]
for i in tok:
    if(i!= None):
       n_token.append(i)
 
c = ''.join(n_token);
print('Test_i/p',c)


# Output data_file

with open('ptb.txt','r') as f:
     contents = f.readlines()
	
	
y=[]
for l in contents[:len(contents)]:
    token=[vocab.get(w) for w in l]
    n_token=[]
    for i in token:
	    if(i!=None):
		   n_token.append(i)
    y.append(n_token)

max_len=0
opf=[]
w=[]
for i in y:
    if max_len<=len(i):
       max_len=len(i)
    else: max_len=max_len   

for i in y:
    j=[]
    j=np.lib.pad(i, (0,max_len-len(i)), 'constant', constant_values=(pad))
    p=np.ones_like(j,dtype=np.float32)
    
    for i in xrange(len(j)):
        if j[i]==pad:
           index_v=i
           p[index_v]=0

    w.append(p)       

	
    opf.append(j)
	
data_y = np.array(opf).T


#train_y

testing_size=int(data_y.shape[1]-1)
train_y=data_y[:,:testing_size]

print('train_y.shape', train_y.shape)


#test_y
test_y=data_y[:,testing_size:]
print('test_y.shape', test_y.shape)



t=test_y.flatten()
tok=[rev_vocab.get(i) for i in t]
     
n_token=[]
for i in tok:
    if(i!= None):
       n_token.append(i)
 
c = ''.join(n_token);
print('Test_o/p',c)


#data_weights

testing_size=int(data_y.shape[1]-1)

data_weight=np.array(w).T
weight=data_weight[:,:testing_size]


print('weight.shape', weight.shape)






with open('new_data.pickle','wb') as f:
	pickle.dump([train_x,train_y,test_x,test_y,weight,seq_length,n_sents,vocb_size,rev_vocab],f)
		
