import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

'''
Aux functions
'''
def _init_matrix(*dims):
   m = torch.Tensor(*dims)
   #from torch.nn.Linear source
   init.kaiming_uniform_(m, a=np.sqrt(5))
   return m


'''
Embeddings
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model

        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        return pos_emb[None,:,:]
         
''' 
Attention building blocks
'''

class MultiHeadedAttention(nn.Module):
   
   def __init__(self, d_model, h, mask=None, dropout=0.1):
      super(MultiHeadedAttention, self).__init__()
      
      self.h  = h
      self.d_model = d_model
      
      self.Wq = nn.Linear(d_model, d_model)
      self.Wk = nn.Linear(d_model, d_model)
      self.Wv = nn.Linear(d_model, d_model)
      self.Wo = nn.Linear(d_model, d_model)
      self.do = nn.Dropout(p = dropout) if dropout > 0 else None
      self.ln = nn.LayerNorm(d_model)

      self.mask = mask
      
   def forward(self, q, v, mask=None):
      ''' 
         q, k, v ~ (Batch, L, d_model)
      Qh, Kh, Vh ~ (Batch, h, d_model/h, L)
              qk ~ (Batch, h, L, L)
              Oh ~ (Batch, h, d_model/h, L)
               O ~ (Batch, L, d_model)
      '''
      Qh = self.Wq(q).transpose(-2,-1).contiguous().view(q.shape[0], self.h, -1, q.shape[-2])
      Kh = self.Wk(q).transpose(-2,-1).contiguous().view(q.shape[0], self.h, -1, q.shape[-2])
      Vh = self.Wv(v).transpose(-2,-1).contiguous().view(q.shape[0], self.h, -1, q.shape[-2])

      # Scaled Dot Product Attention in h blocks QKh_b = dot(Qh_b^T, Kh_b) for all h (head) and b (batch)
      qk = torch.einsum('ijkl,ijkn->ijln', (Qh, Kh))/np.sqrt(self.d_model)

      # Reset mask values to -Inf (softmax prob ~ 0)
      if mask is not None:
         qk = qk.masked_fill(mask == 0, float('-inf'))
      elif self.mask is not None:
            qk = qk.masked_fill(self.mask == 0, float('-inf'))
         
      # Softmax on sample dimension (not d_model)
      p_attn = F.softmax(qk, dim=-1)

      # Dropout to attention probabilities
      if self.do is not None:
         p_attn = self.do(p_attn)

      # Apply attention to Vh -> Oh = dot(p_attn, Vh^T)
      Oh = torch.einsum('ijkl,ijml->ijkm', (Vh, p_attn))

      # Concatenate attention output
      O = Oh.view(Oh.shape[0], -1, Oh.shape[-1]).transpose(-2,-1)

      # Layer norm and residual connection
      return self.ln(q + self.Wo(O))

class RelativeMultiHeadedAttention(nn.Module):

   def __init__(self, L, d_model, h, mask=None, dropout=0.1):
      super(RelativeMultiHeadedAttention, self).__init__()
      assert d_model % h == 0
      self.L = L
      self.h = h
      self.d_model = d_model
      self.mask = mask
      R = PositionalEncoding(d_model)(torch.arange(L-1,-L,-1,dtype=torch.float32)) # Relative positional encodings
      R = R.transpose(-2,-1).contiguous()
      self.register_buffer('R', R)

      # Linear transformations of embeddings
      self.Wq = nn.Parameter(_init_matrix(h, d_model, d_model//h))
      self.Wv = nn.Parameter(_init_matrix(h, d_model//h, d_model))
      self.Wke = nn.Parameter(_init_matrix(h, d_model//h, d_model))
      self.Wkr = nn.Parameter(_init_matrix(h, d_model//h, d_model))

      # Position and content biases
      self.cb = nn.Parameter(_init_matrix(1,h,1,d_model//h)) # Content bias
      self.pb = nn.Parameter(_init_matrix(1,h,1,d_model//h)) # Position bias

      # Output layers
      self.do = nn.Dropout(p = dropout) if dropout > 0 else None      
      self.Wo = nn.Linear(d_model, d_model)
      self.ln = nn.LayerNorm(d_model)
      
   def _shift_b(self, b):
      # Inspired by https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/mem_transformer.py#L194
      z = torch.zeros(b.shape[0], b.shape[1], b.shape[2], 1, device=b.device)
      b_pad = torch.cat((z,b), 3).view(b.shape[0], b.shape[1], b.shape[3]+1, b.shape[2])
      return b_pad[:,:,1:].view_as(b)[:,:,:,:b.shape[2]]

   
   def forward(self, E, Ev, mask=None):
      '''
         Ev, E  ~  (Batch, L, d_model)
             R  ~  (1, d_model, 2L-1)
            Wq  ~  (h, d_model, d_model/h)
    Wv,Wke,Wkr  ~  (h, d_model/h, d_model)
        cb, pb  ~  (1, h, 1, d_model/h)
             q  ~  (Batch, h, L, d_model/h)
          v, k  ~  (Batch, h, d_model/h, L)
             Q  ~  (1, h, d_model/h, 2L-1)
             b  ~  (Batch, h, L, 2L-1)
       A, D, B  ~  (Batch, h, L, L)
            Oh  ~  (Batch, h, d_model/h, L)
             O  ~  (Batch, L, d_model)
      '''
      q = torch.einsum('iykl,xjlm->ijkm', (E[:,None,:,:], self.Wq[None,:,:,:]))
      k = torch.einsum('xjkl,iyml->ijkm', (self.Wke[None,:,:,:], E[:,None,:,:]))
      v = torch.einsum('xjkl,iyml->ijkm', (self.Wv[None,:,:,:], Ev[:,None,:,:]))
      Q = torch.einsum('ijk,xkl->ijl', (self.Wkr, self.R))[None,:,:,:]
      b = torch.einsum('ijkl,xjlm->ijkm', (q, Q))
      d = torch.einsum('ijkl,ijlm->ijkm', (self.pb, Q))
      
      # Attention matrix
      A_a = torch.einsum('ijkl,ijlm->ijkm',(q,k))
      A_b = self._shift_b(b)
      
      # Compute bias vector and replicate to row dimension L
      A_c = torch.einsum('xjkl,ijlm->ijkm',(self.cb,k)).repeat(1,1,k.shape[-1],1)
      A_d = self._shift_b(d.repeat(1,1,E.shape[-2],1))

      # Attention matrix
      A = A_a + A_b + A_c + A_d

      if mask is not None:
         A = A.masked_fill(mask == 0, float('-inf'))
      elif self.mask is not None:
         A = A.masked_fill(self.mask == 0, float('-inf'))

      # Attention softmax
      p_attn = F.softmax(A, dim=-1)

      # Dropout to attention probabilities
      if self.do is not None:
         p_attn = self.do(p_attn)

      # Apply attention to v
      Oh = torch.einsum('ijkl,ijml->ijkm',(v,A))

      # Concatenate attention output
      O = Oh.view(Oh.shape[0], -1, Oh.shape[-1]).transpose(-2,-1)

      # Layer norm and residual connection
      return self.ln(Ev + self.Wo(O))

   
class FeedForwardNet(nn.Module):
   def __init__(self, d_model, d_ffn):
      super(FeedForwardNet, self).__init__()
      
      self.ff = nn.Sequential(nn.Linear(d_model, d_ffn),
                              nn.ReLU(),
                              nn.Linear(d_ffn, d_model))
      self.ln = nn.LayerNorm(d_model)

   def forward(self, x):
      return self.ln(x + self.ff(x))

   
''' 
Encoder and Decoder Blocks
'''

class EncoderBlock(nn.Module):
   def __init__(self, d_model, d_ffn, h, dropout=0.1):
      super(EncoderBlock, self).__init__()
      
      self.attn      = MultiHeadedAttention(d_model, h, dropout=dropout)
      self.ffn       = FeedForwardNet(d_model, d_ffn)
      
   def forward(self, x, mask):
      return self.ffn(self.attn(x,x,mask))


class RelativeEncoderBlock(nn.Module):
   def __init__(self, L, d_model, d_ffn, h, dropout=0.1):
      super(RelativeEncoderBlock, self).__init__()
      
      self.attn      = RelativeMultiHeadedAttention(L, d_model, h, dropout=dropout)
      self.ffn       = FeedForwardNet(d_model, d_ffn)
      
   def forward(self, x, mask):
      return self.ffn(self.attn(x,x,mask))

   
class DecoderBlock(nn.Module):
   def __init__(self, d_model, d_ffn, h, mask, dropout=0.1):
      super(DecoderBlock, self).__init__()
      
      self.attn0      = MultiHeadedAttention(d_model, h, mask=mask, dropout=dropout)
      self.attn1      = MultiHeadedAttention(d_model, h, mask=None, dropout=dropout)
      self.ffn        = FeedForwardNet(d_model, d_ffn)
      
   def forward(self, x, vk):
      y  = self.attn0(x,x)
      z  = self.attn1(y,vk)
      return self.ffn(z)

class RelativeDecoderBlock(nn.Module):
   def __init__(self, L, d_model, d_ffn, h, mask, dropout=0.1):
      super(RelativeDecoderBlock, self).__init__()
      
      self.attn0      = RelativeMultiHeadedAttention(L, d_model, h, mask=mask, dropout=dropout)
      self.attn1      = RelativeMultiHeadedAttention(L, d_model, h, mask=None, dropout=dropout)
      self.ffn        = FeedForwardNet(d_model, d_ffn)
      
   def forward(self, x, vk):
      y  = self.attn0(x,x)
      z  = self.attn1(y,vk)
      return self.ffn(z)


'''
ALBERT
'''

class ALBERT(nn.Module):
   def __init__(self, N, E, H, h, d_ffn, L, n_word, dropout=0.1, device=None):
      super(ALBERT,self).__init__()

      # Model parameters
      self.N = N
      self.E = E
      self.H = H
      self.h = h
      self.d_ffn = d_ffn
      # Text parameters
      self.L = L
      self.n_word
      # Model device
      self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

      # Input Embedding Transformations
      self.wrd_emb = nn.Sequential(nn.Embedding(n_word, E), nn.Linear(E, H)).to(self.device)

      # Self-attention layers
      self.attn    = [attention.RelativeEncoderBlock(L, H, d_ffn, h, dropout=dropout)].to(self.device) for _ in range(N)]

      # Output softmax
      self.outprob = nn.Sequential(nn.Linear(H, n_word), nn.Softmax(-1)).to(self.device)
      

   def forward(self, seq, mask):
      x = self.wrd_emb(seq)
      for att in self.attn:
         x = att(x, mask)
      return self.outprob(x)
      

class BertMLMHead(nn.Module):
   def __init__(self):
      super(BertMLMHead, self).__init__()
      # Layers:
      # Last attention output of BERT (size H).
      # Linear HxH
      # Activation
      # LayerNorm
      # Embedding Output (Linear+Bias), optionally Linear layer is weight-tied to input Embedding
      
