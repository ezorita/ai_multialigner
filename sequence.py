import torch
import torch.nn as nn
import numpy as np

class ProteinSeq():
   def __init__(self, fname, test_size=0.2, random_state=None, device='cpu'):
      # Number of symbols
      self.n_symbols = 21

      # Residue indices
      self.RESIDUE_IDX = {'X': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
                          'H': 7, 'I': 8, 'K': 9, 'L': 10,'M': 11,'N': 12,'P': 13,
                          'Q': 14,'R': 15,'S': 16,'T': 17,'V': 18,'W': 19,'Y': 20
      }

      
      # Preallocate aminoacid table
      to_idx = lambda x: [self.RESIDUE_IDX[r] for r in x]

      # Read raw file
      with open(fname) as f:
         raw = f.read().split('\n')
         
      # Convert residues to indices
      seqs = [torch.LongTensor(to_idx(x)) for x in raw][:-1]
      del raw

      # Pad sequences, convert to tensor and send to device
      self.idseq = nn.utils.rnn.pad_sequence(seqs).transpose(0,1).to(device)
      del seqs

      self.seqlen = self.idseq.shape[1]
         
      # Train/test split
      idx = np.arange(self.idseq.shape[0])
      if random_state:
         np.random.seed(random_state)
      np.random.shuffle(idx)
      test_size = int(np.round(self.idseq.shape[0]*test_size))
      self.test_seq  = self.idseq[idx[:test_size]]
      self.train_seq = self.idseq[idx[test_size:]]
      del self.idseq

   def random_train_batch(self, batch_size):
      return self.train_seq[np.random.choice(self.train_seq.shape[0], size=batch_size, replace=False)]

   def random_test_batch(self, batch_size):
      return self.test_seq[np.random.choice(self.test_seq.shape[0], size=batch_size, replace=False)]

   def train_batches(self, batch_size):
      idx = np.arange(self.train_seq.shape[0])
      np.random.shuffle(idx)
      return [self.train_seq[i] for i in np.array_split(idx, self.train_seq.shape[0]//batch_size)]

   def test_batches(self, batch_size):
      idx = np.arange(self.test_seq.shape[0])
      np.random.shuffle(idx)
      return [self.test_seq[i] for i in np.array_split(idx, self.test_seq.shape[0].shape[0]//batch_size)]


# Applies random deletions, returns new sequence and attention mask
def random_dels(seq, n_del_func):
   # Create list of sequences with deletions
   dseqs = [x[np.sort(np.random.choice(seq.shape[1],size=seq.shape[1] - n_del_func(),replace=False))] for x in seq]
   # Extend maximum length to original
   dseqs.append(torch.zeros(seq.shape[1], dtype=seq.dtype, device=seq.device))
   # Pad again to full length
   return nn.utils.rnn.pad_sequence(dseqs).transpose(0,1)[:-1]

   
# Converts a index sequence to one-hot
def to_onehot(seq, n_symbols, dtype=None):
   seq_oh = torch.zeros((seq.shape[0]*seq.shape[1], n_symbols), dtype=seq.dtype if not dtype else dtype, device=seq.device)
   seq_oh.scatter_(1, seq.view(-1,1), 1)
   return seq_oh.view(seq.shape[0], seq.shape[1], -1)
