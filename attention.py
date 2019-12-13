import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
      
   def forward(self, q, k, v, mask=None):
      ''' 
         q, k, v ~ (Batch, L, d_model)
      Qh, Kh, Vh ~ (Batch, h, d_model/h, L)
              qk ~ (Batch, h, L, L)
              Oh ~ (Batch, h, d_model/h, L)
               O ~ (Batch, L, d_model)
      '''
      Qh = self.Wq(q).transpose(-2,-1).contiguous().view(q.shape[0], self.h, -1, q.shape[-2])
      Kh = self.Wk(k).transpose(-2,-1).contiguous().view(q.shape[0], self.h, -1, q.shape[-2])
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
      return self.ffn(self.attn(x,x,x,mask))

   
class DecoderBlock(nn.Module):
   def __init__(self, d_model, d_ffn, h, mask, dropout=0.1):
      super(DecoderBlock, self).__init__()
      
      self.attn0      = MultiHeadedAttention(d_model, h, mask=mask, dropout=dropout)
      self.attn1      = MultiHeadedAttention(d_model, h, mask=None, dropout=dropout)
      self.ffn        = FeedForwardNet(d_model, d_ffn)
      
   def forward(self, x, vk):
      y  = self.attn0(x,x,x)
      z  = self.attn1(y,vk,vk)
      return self.ffn(z)

