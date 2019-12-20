import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

# Local modules
import sequence
import attention
from lookahead import Lookahead

# Avoid plot crash without X-server
plt.switch_backend('agg')

   
'''
Transformer
'''

class NoiseAttention(nn.Module):
   def __init__(self, L, n_word, d_model, n_att, d_ffn, h, dropout=0.1, device=None, relative=True):
      super(NoiseAttention, self).__init__()

      # Parameters
      self.L = L
      self.n_word = n_word
      self.d_model = d_model
      self.h = h
      self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
      self.relative = relative

      # Input Embedding Transformations
      self.src_emb = nn.Embedding(n_word, d_model).to(self.device)
      self.emb_wgt = None
      self.pos_enc = None

      if not relative:
         # Positional encodings
         self.pos_enc = attention.PositionalEncoding(d_model)(torch.arange(L, dtype=torch.float32)).to(self.device)
         self.emb_wgt = np.sqrt(d_model)
         self.selfattn = [attention.EncoderBlock(d_model, d_ffn, h, dropout=dropout).to(self.device) for _ in range(n_att)]
      else:
         # Instantiate Encoder and Decoder blocks
         self.selfattn = [attention.RelativeEncoderBlock(L, d_model, d_ffn, h, dropout=dropout).to(self.device) for _ in range(n_att)]
      

      # Output Softmax
      self.outprob = nn.Sequential(nn.Linear(d_model, n_word), nn.Softmax(-1)).to(self.device)

      
   def forward(self, seq, mask):
      E = self.src_emb(seq)
      if not self.relative:
         E = E*self.emb_wgt + self.pos_enc
      for attn in self.selfattn:
         x = attn(E, mask)
      return self.outprob(x)

   
   def optimize(self, seqs, epochs, batch_size, ndel_func, init_lr=.01, min_lr=.001, lr_exp=1.01):

      # Optimizer
      base_opt = torch.optim.Adam(self.parameters(), lr=init_lr**(1/lr_exp))
      opt = Lookahead(base_opt, k=5, alpha=0.5)
      opt = base_opt
      #opt_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=10, verbose=True)

      # Logs
      train_loss = np.zeros(epochs)
      test_loss  = np.zeros(epochs)

      for e in torch.arange(epochs):
         # Update LR
         cur_lr = base_opt.param_groups[0]['lr']
         if cur_lr > min_lr:
            base_opt.param_groups[0]['lr'] = max(min_lr,cur_lr**lr_exp)

         for x in seqs.train_batches(batch_size=batch_size):
            # Compute one-hot encoded targets
            target = sequence.to_onehot(x, seqs.n_symbols, dtype=torch.float32)
            # Apply random deletions
            xdel = sequence.random_dels(x, ndel_func)
            # Attention
            y = self(xdel, None)
            loss = torch.nn.functional.binary_cross_entropy(y, target)
            mem = torch.cuda.memory_allocated()/1024/1024

            # Update parameters
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            l = float(loss.detach().to('cpu'))
            train_loss[e] += l/(seqs.train_seq.shape[0]//batch_size)

         # Test data
         with torch.no_grad():
            # Generate test batch
            x = seqs.random_test_batch(batch_size=10*batch_size)
            # Compute one-hot encoded targets
            target = sequence.to_onehot(x, seqs.n_symbols, dtype=torch.float32)
            # Apply random deletions
            xdel = sequence.random_dels(x, ndel_func)
            # Attention
            y = self(xdel, None)
            test_loss[e] = float(torch.nn.functional.binary_cross_entropy(y, target).to('cpu'))

         # Optimizer lr schedule
         #opt_sched.step(train_loss[e])

         # Verbose epoch
         print("[epoch {}] train_loss={:.3f} test_loss={:.3f} memory={:.2f}MB".format(e+1, 100*train_loss[:e+1].mean(), 100*test_loss[:e+1].mean(), mem))

         # Save model
         if (e+1)%10 == 0:
            torch.save(self, 'saved_models/delattn_model_epoch{}.torch'.format(e))
            

      return train_loss, test_loss
      


if __name__ == "__main__":

   if sys.argv[1] == 'train':
      seqs = sequence.ProteinSeq(sys.argv[2], test_size=0.2, device='cuda')
      # Instantiate NoiseAttention
      attmodel = NoiseAttention(
         L = seqs.seqlen,
         n_word = seqs.n_symbols,
         d_model = 256,
         n_att = 4,
         d_ffn = 1024,
         h = 8,
         relative=True,
         device = 'cuda'
      )

      epochs = 300
      batch_size = 16
      init_lr = .01
      min_lr =.001
      lr_exp = 1.01
      ndel = lambda: np.random.randint(0,8)
      
      train_loss, test_loss = attmodel.optimize(seqs, epochs=epochs, batch_size=batch_size, ndel_func=ndel, init_lr=init_lr, min_lr=min_lr)
      torch.save(attmodel, 'delattn_model.torch')
      
      plt.figure()
      plt.plot(train_loss)
      plt.plot(test_loss)
      plt.legend(["Train", "Test"])
      plt.xlabel("epoch")
      plt.ylabel("BCE loss")
      plt.savefig('loss.png')

   elif sys.argv[1] == 'eval':
      gapatt = torch.load('delattn_model.torch')
      x = torch.LongTensor(data[0,:]).cuda()
      gapatt(x, None)

