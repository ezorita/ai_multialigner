import torch
import torch.nn as nn
import sys
import numpy as np
import pdb

cnn_out = lambda i,k,s: (i-k)//s+1

class SeqCNN(nn.Module):
   def __init__(self, seq_len, n_aa, kernel=5, channels=32, cnn_layers=4, enc_layers=2, enc_units=1024, lat_sz=100):
      super(SeqCNN, self).__init__()

      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

      ## CNN Network
      ## Several CNN passes followed by (4,2) downsamplers

      cnn_net = [
         nn.ReflectionPad2d(0,0,kernel//2, kernel//2),
         nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(kernel, n_aa)),
         nn.LeakyReLU()
      ]

      for i in np.arange(cnn_layers):
         cnn_net.extend([
            nn.ReflectionPad2d(0,0,kernel//2,kernel//2),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU()
         ])

      # Compute number of downsamplers
      n_down = 0
      out_sz = seq_len
      while out_sz > kernel:
         out_sz = cnn_out(out_sz,4,2)
         n_down += 1
      
      for i in np.arange(n_down):
         cnn_net.extend([
            nn.Conv1d(in_channels=channels*(i**2), out_channels=channels*((i+1)**2), kernel_size=4, stride=2),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU()
         ])

      if out_sz > 1:
         cnn_net.extend([
            nn.Conv1d(in_channels=channels*(n_down**2), out_channels=channels*(n_down**2), kernel_size=out_sz),
            nn.LeakyReLU()
         ])

      self.cnn = nn.Sequential(**cnn_net).to(self.device)

      ## Encoder network
      enc_net = [
         nn.Linear(channels*n_down**2, enc_units),
         nn.ReLU()
      ]

      for i in np.arange(enc_layers - 1):
         enc_net.extend([
            nn.Linear(enc_units, enc_units),
            nn.ReLU()
         ])

      enc_net.extend([
         nn.Linear(enc_units, lat_sz)
      ])

      self.encoder = nn.Sequential(**enc_net).to(self.device)

            
   def forward(x):
      # Encoder
      y = self.cnn(x)
      z = self.encoder(y[:,:,0])

      # Decoder
      # ...

      return z
      

if __name__ == "__main__":

   # Preallocate aminoacid table
   n_aa = 23
   onehot = np.eye(n_aa)
   to_oh = {
      'A': onehot[0],
      'B': onehot[1],
      'C': onehot[2],
      'D': onehot[3],
      'E': onehot[4],
      'F': onehot[5],
      'G': onehot[6],
      'H': onehot[7],
      'I': onehot[8],
      'K': onehot[9],
      'L': onehot[10],
      'M': onehot[11],
      'N': onehot[12],
      'P': onehot[13],
      'Q': onehot[14],
      'R': onehot[15],
      'S': onehot[16],
      'T': onehot[17],
      'V': onehot[18],
      'W': onehot[19],
      'X': onehot[20],
      'Y': onehot[21],
      'Z': onehot[22]
   }


   # Load sequence data
   fasta_file = sys.argv[1]
   
   # Read file
   seqs = []
   with open(fasta_file) as f:
      seq = ''
      for line in f:
         if line[0] == '>':
            if seq:
               seqs.append(seq)
            seq = ''
         else:
            seq += line.rstrip()

   lens = np.unique(np.array([len(s) for s in seqs]), return_counts=True)
   seq_cnt = np.max(lens[1])
   seq_len = lens[0][np.argmax(lens[1])]
   
   # Allocate sequence tensor
   data = torch.zeros(seq_cnt, seq_len, n_aa)
   i = 0
   for s in seqs:
      if len(s) != seq_len:
         continue
      else:
         data[i] = torch.Tensor([to_oh[c] for c in s])
   pdb.set_trace()   
