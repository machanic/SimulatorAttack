from torch import nn
import torch
import numpy as np
from torch.nn import functional as F


class NP(nn.Module):
    def __init__(self, c, device, data_dim, args):
        super(NP, self).__init__()
        self.r_dim = args.r_dim
        self.z_dim = args.z_dim
        self.hidden_dim = args.hidden_dim
        self.device = device
        self.data_dim = data_dim
        self.h_1 = nn.Linear(c + 1, self.hidden_dim)
        self.h_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h_3 = nn.Linear(self.hidden_dim, self.r_dim)

        self.s_dense = nn.Linear(self.r_dim, self.r_dim)

        self.s_to_z_mean = nn.Linear(self.r_dim, self.z_dim)
        self.s_to_z_logvar = nn.Linear(self.r_dim, self.z_dim)

        self.h_4 = nn.Linear(c + 1, self.hidden_dim)
        self.h_5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h_6 = nn.Linear(self.hidden_dim, self.r_dim)
        self.s = nn.Linear(self.r_dim, self.z_dim)

        self.h_7 = nn.Linear(c, self.hidden_dim)
        self.h_8 = nn.Linear(self.hidden_dim, c)

        self.h_9 = nn.Linear(c, self.hidden_dim)
        self.h_10 = nn.Linear(self.hidden_dim, c)

        if args.rep_type == 'mlp':
            self.att_rep_k = nn.Linear(self.z_dim, self.z_dim)
            self.att_rep_k = nn.Linear(self.z_dim, self.z_dim)

        self.num_heads = 4

        if args.att_type == 'multihead':
            d_k, d_v = c, self.r_dim
            self.heads = []
            head_size = int(self.r_dim / self.num_heads)

            for i in range(self.num_heads):
                self.heads.append(nn.Conv1d(d_k, d_k, 1))
                self.heads[-1].weight.data.normal_(0, d_k ** (-0.5))
                self.heads.append(nn.Conv1d(d_k, d_k, 1))
                self.heads[-1].weight.data.normal_(0, d_k ** (-0.5))
                self.heads.append(nn.Conv1d(d_v, head_size, 1))
                self.heads[-1].weight.data.normal_(0, d_k ** (-0.5))
                self.heads.append(nn.Conv1d(head_size, self.r_dim, 1))
                self.heads[-1].weight.data.normal_(0, d_v ** (-0.5))

            self.heads = nn.Sequential(*(self.heads))

        self.g_1 = nn.Linear(self.z_dim * 2 + c, self.hidden_dim)
        self.g_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.g_3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.g_4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.g_5 = nn.Linear(self.hidden_dim, c)
        self.g_mean = nn.Linear(self.hidden_dim, 1)
        self.g_var = nn.Linear(self.hidden_dim, 1)

        self.attention_type = args.att_type

    def encoder_determinate(self, x_y):
        x_y = F.relu(self.h_1(x_y))
        x_y = F.relu(self.h_2(x_y))
        x_y = F.relu(self.h_3(x_y))
        return x_y

    def crossAtt_x(self, x):
        # print('x-----------------',x.shape)
        x = F.relu(self.h_7(x))
        # print('x-----------------',x.shape)
        x = F.relu(self.h_8(x))
        # print('x-----------------',x.shape)
        return x

    def crossAtt_xTarget(self, x_target):
        x_target = F.relu(self.h_9(x_target))
        x_target = F.relu(self.h_10(x_target))
        return x_target

    def encoder_latent(self, x_y):
        # print('h--------------',x_y.shape)
        x_y = F.relu(self.h_4(x_y))
        # print('h_4---------------',x_y.shape)
        x_y = F.relu(self.h_5(x_y))
        # print('h_5---------------',x_y.shape)
        x_y = F.relu(self.h_6(x_y))
        # print('h_6---------------',x_y.shape)
        return x_y

    def dot_product_attention(self, q, k, v, normalise):
        """Computes dot product attention.

        Args:
          q: queries. tensor of  shape [B,m,d_k].
          k: keys. tensor of shape [B,n,d_k].
          v: values. tensor of shape [B,n,d_v].
          normalise: Boolean that determines whether weights sum to 1.

        Returns:
          tensor of shape [B,m,d_v].
        """
        # print (q.shape, k.shape, v.shape)
        d_k = q.shape[-1]
        scale = np.sqrt(1.0 * d_k)
        unnorm_weights = torch.einsum('bjk,bik->bij', (k, q)) / scale  # [B,m,n]
        if normalise:
            weight_fn = nn.functional.softmax
        else:
            weight_fn = torch.sigmoid

        weights = weight_fn(unnorm_weights, dim=-1)  # [B,m,n]
        rep = torch.einsum('bik,bkj->bij', (weights, v))  # [B,m,d_v]
        return rep

    def laplace_attention(self, q, k, v, scale, normalise):
        """Computes laplace exponential attention.

        Args:
          q: queries. tensor of shape [B,m,d_k].
          k: keys. tensor of shape [B,n,d_k].
          v: values. tensor of shape [B,n,d_v].
          scale: float that scales the L1 distance.
          normalise: Boolean that determines whether weights sum to 1.

        Returns:
          tensor of shape [B,m,d_v].
        """
        k = k.unsqueeze(dim=1)  # [B,1,n,d_k]
        q = q.unsqueeze(dim=2)  # [B,m,1,d_k]
        unnorm_weights = -((k - q) / scale).abs()  # [B,m,n,d_k]
        unnorm_weights = unnorm_weights.sum(dim=-1)  # [B,m,n]
        if normalise:
            weight_fn = nn.functional.softmax
        else:
            weight_fn = lambda x: 1 + nn.functional.tanh(x)
        weights = weight_fn(unnorm_weights, dim=-1)  # [B,m,n]
        rep = torch.einsum('bik,bkj->bij', (weights, v))  # [B,m,d_v]
        return rep

    def multihead_attention(self, q, k, v, num_heads=8):
        """Computes multi-head attention.

        Args:
          q: queries. tensor of  shape [B,m,d_k].
          k: keys. tensor of shape [B,n,d_k].
          v: values. tensor of shape [B,n,d_v].
          num_heads: number of heads. Should divide d_v.

        Returns:
          tensor of shape [B,m,d_v].
        """
        # print (q.shape)
        # print (k.shape)
        # print (v.shape)
        d_k = q.shape[-1]
        d_v = v.shape[-1]
        head_size = d_v / num_heads

        rep = 0.0
        for h in range(num_heads):
            o = self.dot_product_attention(
                self.heads[h * 4](q.permute(0, 2, 1)).permute(0, 2, 1),
                self.heads[h * 4 + 1](k.permute(0, 2, 1)).permute(0, 2, 1),
                self.heads[h * 4 + 2](v.permute(0, 2, 1)).permute(0, 2, 1),
                normalise=True)

            rep += self.heads[h * 4 + 3](o.permute(0, 2, 1)).permute(0, 2, 1)
        return rep

    def corss_attention(self, content_x, target_x, r):
        if self.attention_type == 'uniform':
            return torch.mean(r, dim=1).unsqueeze(1).expand(-1, self.data_dim, -1)
        elif self.attention_type == 'laplace':
            return self.laplace_attention(target_x, content_x, r, 1, True)
        elif self.attention_type == 'dot':
            return self.dot_product_attention(target_x, content_x, r, True)
        elif self.attention_type == 'multihead':
            return self.multihead_attention(target_x, content_x, r, self.num_heads)

    def self_attention(self, input_xy):
        """
            inputs :
                input_xy : input feature maps [B,C,W,H]
            returns :
                out : self attention value + input feature
                attention: [B,N,N,N](N is Width*Height)
        """
        #  print(input_xy.shape)
        input_xy = input_xy.permute(0, 2, 1)
        # print('input_xy=-----------------',input_xy.shape)
        m_batchsize, C, width_height = input_xy.size()
        q_conv = nn.Conv1d(in_channels=C, out_channels=C // 8, kernel_size=1)
        k_conv = nn.Conv1d(in_channels=C, out_channels=C // 8, kernel_size=1)
        v_conv = nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1)
        gamma = nn.Parameter(torch.zeros(1)).to(self.device)

        q_conv = q_conv.to(self.device)
        k_conv = k_conv.to(self.device)
        v_conv = v_conv.to(self.device)
        q = q_conv(input_xy)
        # print('q-------------------',q.shape)
        q = q.view(m_batchsize, -1, width_height).permute(0, 2, 1)
        k = k_conv(input_xy).view(m_batchsize, -1, width_height)
        energy = torch.bmm(q, k)
        attention = F.softmax(energy, dim=-1)
        v = v_conv(input_xy).view(m_batchsize, -1, width_height)

        rep = torch.bmm(v, attention.permute(0, 2, 1))
        rep = rep.view(m_batchsize, width_height, C)
        rep = rep.permute(0, 2, 1)
        # print('rep--------------',rep.device)
        # print('input_xy---------------------',input_xy.device)
        # print('gamma--------------------',gamma.device)
        rep = gamma * rep + input_xy

        rep = rep.permute(0, 2, 1)
        return rep, attention

    def reparameterise(self, z):
        mu, var = z

        m = torch.distributions.normal.Normal(mu, var)
        z_sample = m.rsample()

        z_sample = z_sample.unsqueeze(1).expand(-1, self.data_dim, -1)

        return z_sample

    def decoder(self, r_sample, z_sample, x_target):  # decoder
        # print (r_sample.shape)
        # print (z_sample.shape)
        # print (x_target.shape)
        z_x = torch.cat([r_sample, z_sample, x_target], dim=2)

        input = F.relu(self.g_1(z_x))
        input = F.relu(self.g_2(input))
        input = F.relu(self.g_3(input))
        input = F.relu(self.g_4(input))

        y_mean = self.g_mean(input)
        y_var = self.g_var(input)
        sigma = 0.1 + 0.9 * F.softplus(y_var)

        y_dis = torch.distributions.normal.Normal(y_mean, sigma)

        y_hat = y_mean
        return (y_hat, y_dis)

    def xy_to_z(self, x, y):  # latent path of encoder
        # print('x--------------',x.shape)
        # print('y--------------',y.shape)
        x_y = torch.cat([x, y], dim=2)
        # print(x_y.shape)
        input_xy = self.encoder_latent(x_y)
        # print(input_xy.shape)
        s_i, _ = self.self_attention(input_xy)
        s = torch.mean(s_i, dim=1)

        s = F.relu(self.s_dense(s))
        mu = self.s_to_z_mean(s)
        logvar = self.s_to_z_logvar(s)
        var = (0.1 + 0.9 * torch.sigmoid(logvar)) * 0.1

        return mu, var

    def xy_to_r(self, x, y, x_target):  # deterministic path of encoder
        x_y = torch.cat([x, y], dim=2)
        input_xy = self.encoder_determinate(x_y)
        r_i, _ = self.self_attention(input_xy)

        x = self.crossAtt_x(x)
        x_target = self.crossAtt_xTarget(x_target)
        r = self.corss_attention(x, x_target, r_i)

        return self.s(r)

    def forward(self, x_context, y_context, x_grid, x_all=None, y_all=None):
        x_target = x_grid.expand(y_context.shape[0], -1, -1)

        z_context = self.xy_to_z(x_context, y_context)  # (mu, logvar) of z
        # print (z_context.shape)
        r_context = self.xy_to_r(x_context, y_context, x_target)

        if self.training:  # loss function will try to keep z_context close to z_all
            z_all = self.xy_to_z(x_all, y_all)
            r_all = r_context
        else:  # at test time we don't have the image so we use only the context
            z_all = z_context
            r_all = r_context

        z_sample = self.reparameterise(z_all)
        r_sample = r_all

        # reconstruct the whole image including the provided context points
        y_hat = self.decoder(r_sample, z_sample, x_target)

        return y_hat, z_all, z_context, r_sample, z_sample