import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import reshape_last_dim

class MDN(nn.Module):
    def __init__(self, dim_input, dim_output, num_components):
        super(MDN, self).__init__()
        self.num_components = num_components
        self.fc_pi = nn.Linear(dim_input, num_components)
        self.fc_mu = nn.Linear(dim_input, num_components * dim_output)
        self.fc_sigma = nn.Linear(dim_input, num_components * dim_output)

    def forward(self, x):
        pi = F.softmax(self.fc_pi(x), -1)
        mu = reshape_last_dim(self.fc_mu(x), self.num_components, -1)
        sigma = reshape_last_dim(torch.exp(self.fc_sigma(x)), self.num_components, -1)
        return pi, mu, sigma
    
    def sample(self, x, argmax=False):
        pi, mu, sigma = self(x)
        if argmax:
            pis = torch.argmax(pi, -1).unsqueeze(-1).unsqueeze(-1)
        else:
            pis = torch.distributions.Categorical(pi).sample().unsqueeze(-1).unsqueeze(-1)
        samples = torch.gather(mu, -2, pis.repeat(1, 1, 1, mu.size(-1)))
        if not argmax:
            samples = samples + torch.randn_like(samples) * torch.gather(sigma, -2, pis.repeat(1, 1, 1, sigma.size(-1)))
        samples = samples.squeeze(-2)
        return samples
    
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2) + Q
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=1, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer2(nn.Module):
    def __init__(self, dim_input, dim_hidden, num_heads, ln=False):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-2)