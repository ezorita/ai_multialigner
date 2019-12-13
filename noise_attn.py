import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import pdb

# Local modules
import sequence
import attention

# Avoid plot crash without X-server
plt.switch_backend('agg')

   
'''
Transformer
'''

class NoiseAttention(nn.Module):
   def __init__(self, L, n_word, d_model, n_att, d_ffn, h, dropout=0.1, device=None):
      super(NoiseAttention, self).__init__()

      # Parameters
      self.L = L
      self.n_word = n_word
      self.d_model = d_model
      self.h = h
      self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

      # Input Embedding Transformations
      self.src_emb = nn.Embedding(n_word, d_model).to(self.device)
      self.emb_wgt = np.sqrt(d_model)

      # Positional encodings
      self.pos_enc  = attention.PositionalEncoding(d_model)(torch.arange(L, dtype=torch.float32)).to(self.device)
      
      # Instantiate Encoder and Decoder blocks
      self.selfattn = [attention.EncoderBlock(d_model, d_ffn, h, dropout).to(self.device) for _ in range(n_att)]

      # Output Softmax
      self.outprob = nn.Sequential(nn.Linear(d_model, n_word), nn.Softmax(-1)).to(self.device)

      # Training status
      self.steps = 0

      
   def forward(self, seq, mask):
      E = self.src_emb(seq)
      x = E*self.emb_wgt + self.pos_enc
      for attn in self.selfattn:
         x = attn(x, mask)
      return self.outprob(x)

   
   def optimize(self, seq, epochs, batch_size, ndel_func):

      # Optimizer
      opt = torch.optim.Adam(self.parameters(), lr=0.001)
      opt_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=10, verbose=True)

      # Logs
      train_loss = np.zeros(epochs)
      test_loss  = np.zeros(epochs)
      
      for e in torch.arange(epochs):
         for x in seq.train_batches(batch_size=batch_size):
            # Compute one-hot encoded targets
            target = sequence.to_onehot(x, seqs.n_symbols, dtype=torch.float32)
            # Apply random deletions
            xdel = sequence.random_dels(x, ndel_func)
            # Attention
            y = self(xdel, None)
            loss = torch.nn.functional.binary_cross_entropy(y, target)

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
         opt_sched.step(train_loss[e])

         # Verbose epoch
         print("[epoch {}] train_loss={} test_loss={}".format(e+1, 100*train_loss[:e+1].mean(), 100*test_loss[:e+1].mean()))

      return train_loss, test_loss
      


if __name__ == "__main__":

   if sys.argv[1] == 'train':
      seqs = sequence.ProteinSeq(sys.argv[2], test_size=0.2, device='cuda')
      # Instantiate NoiseAttention
      attmodel = NoiseAttention(
         L = seqs.seqlen,
         n_word = seqs.n_symbols,
         d_model = 64,
         n_att = 4,
         d_ffn = 256,
         h = 8,
        device = 'cuda'
      )

      epochs = 300
      epochs = 10
      batch_size = 16
      ndel = lambda: np.random.randint(10,30)
      
      train_loss, test_loss = attmodel.optimize(seqs, epochs=epochs, batch_size=batch_size, ndel_func=ndel)
      
      torch.save(gapatt, 'delattn_model.torch')
      
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

