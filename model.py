import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import torch.optim as optim
def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t.cpu())
    return c.reshape(-1, 1, 1, 1).to(consts.device)
class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))


class spatialAttentionGCN(nn.Module):
    def __init__(self, d_model,dropout=.0):
        super(spatialAttentionGCN, self).__init__()
        self.d_model=d_model
        self.Theta = nn.Linear(d_model, d_model, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x ,norm_adj):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        spatial_attention = self.SAt(x) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)

        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        return F.relu(self.Theta(torch.matmul(norm_adj.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.d_model)).transpose(1, 2))
        # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)

def sin_cos_time_embed(ti, d_model):
    """
    使用正弦/余弦编码将时间信息嵌入到高维空间

    Args:
        ti: 时间信息张量，形状为 [batch_size, seq_len] 或 [seq_len]
        d_model: 输出嵌入维度
    Returns:
        嵌入后的时间信息，形状为 [batch_size, seq_len, d_model] 或 [seq_len, d_model]
    """
    if ti.dim() == 1:
        ti = ti.unsqueeze(0)  # [1, seq_len]
    batch_size, seq_len = ti.shape

    # 创建频率波段 [d_model//2]
    freqs = torch.exp(torch.arange(0, d_model//2, dtype=torch.float, device=ti.device) * 
            -(math.log(10000.0) / (d_model//2 - 1)))

    # 计算角度 [batch_size, seq_len, d_model//2]
    angles = ti.unsqueeze(-1) * freqs.view(1, 1, -1)

    # 生成正弦和余弦编码
    sin_enc = torch.sin(angles)  # [batch_size, seq_len, d_model//2]
    cos_enc = torch.cos(angles)  # [batch_size, seq_len, d_model//2]

    # 拼接正弦和余弦编码
    embedded = torch.cat([sin_enc, cos_enc], dim=-1)  # [batch_size, seq_len, d_model]

    # 处理奇数维度情况
    if embedded.size(-1) != d_model:
        if embedded.size(-1) > d_model:
            embedded = embedded[..., :d_model]
        else:
            zeros = torch.zeros(batch_size, seq_len, d_model - embedded.size(-1), 
                             device=embedded.device)
            embedded = torch.cat([embedded, zeros], dim=-1)

    # 如果输入是1维，去掉batch维度
    if ti.dim() == 1:
        embedded = embedded.squeeze(0)

    return embedded

def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)

class MultiHeadAttentionAwareTemporal(nn.Module):
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=0.0):
        super(MultiHeadAttentionAwareTemporal, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.padding = (kernel_size - 1) // 2

        
        self.compute_query=nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding))
        
        self.compute_key=nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding))
        
        self.linear1 = nn.Linear(d_model, d_model)
        
        self.linear2 = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        '''
        :param x: (batch, N, T, d_model) - 
        
        used for query, key, and value 32, 64, 50, 36
        
        new B,d_model,N,T
        :param mask: (batch, T, T)
        :return: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T)

        nbatches = x.size(0)
        N = x.size(2)

        # Apply temporal convolution to query and key (both from x)
        query=self.compute_query(x)
        query=query.view(nbatches, self.h, self.d_k, N, -1).permute(0,3,1,4,2) #32, 8,8, 50, 36
        
        key=self.compute_key(x)
        key=key.view(nbatches, self.h, self.d_k, N, -1).permute(0,3,1,4,2) #32, 8,8, 50, 36

        # Linear projection for value (from x)
        value = self.linear1(x.permute(0,2,3,1)).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) #torch.Size([32, 50, 8, 36, 8])
        # Apply attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # Combine heads
        x = x.transpose(2, 3).contiguous()  # (batch, N, T, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T, d_model)
        return self.linear2(x)

class PredictLayer(nn.Module):
    def __init__(self, hidden_channels, heads):
        super(PredictLayer, self).__init__()
        self.time_attn = MultiHeadAttentionAwareTemporal(heads,hidden_channels)
        self.node_attn = spatialAttentionGCN(hidden_channels)
#         if residual_connection or use_LayerNorm:
#             self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
        self.norm = nn.ModuleList([
            nn.LayerNorm(hidden_channels) 
            for _ in range(2)  
        ])
    def forward(self, x, support_matrix): #32, 64, 50, 36
        '''
        :param x: src: (batch_size, F_in,N, T_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        x = x.permute(0,2,3,1)+self.norm[0](self.time_attn(x))
        
        return x+self.norm[1](self.node_attn(x,support_matrix))


def get_laplacian(graph, I, normalize=True):

    if normalize:
        D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
        #L = I - torch.matmul(torch.matmul(D, graph), D)
        L = torch.matmul(torch.matmul(D, graph), D)
    else:
        graph = graph + I
        D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
        L = torch.matmul(torch.matmul(D, graph), D)
    return L

class Encoder(nn.Module):

    def __init__(self,lap_pos_enc,in_len,out_len, in_channels, hidden_channels, out_channels, dropout,  heads = 8, support_len = 3):
        super(Encoder,self).__init__()

        self.lap_pos_enc=lap_pos_enc
        self.node_nums=self.lap_pos_enc.shape[0]
#         self.node_embed = nn.Embedding(lap_pos_enc.shape[0], embedding_dim=hidden_channels)
        self.node_embed=nn.Linear(hidden_channels,hidden_channels)
        self.convs = nn.ModuleList()
        self.hidden_channels=hidden_channels
        self.in_channels=in_channels
        self.x_e = nn.Linear(in_channels+1,hidden_channels)
#         self.prediction_generator1=nn.Linear(hidden_channels,out_channels)
#         self.prediction_generator2=nn.Linear(in_len,out_len)
        self.num_layers=2
        self.predictlayers = nn.ModuleList([
            PredictLayer(hidden_channels, heads) 
            for _ in range(self.num_layers)  
        ])
        self.fc=nn.Sequential( 
                OrderedDict([('fc1', nn.Linear(hidden_channels, 2*hidden_channels)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(2*hidden_channels, 2*hidden_channels)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(2*hidden_channels, hidden_channels))]))
        self.node_embeddings1=nn.Parameter(torch.randn(self.node_nums, hidden_channels), requires_grad=True)
        self.time_attn1 = MultiHeadAttentionAwareTemporal(heads,hidden_channels)
#         self.predict_layers=PredictLayer(hidden_channels,heads)
        self.norm1 =  nn.LayerNorm(hidden_channels) 
        self.norm2 =  nn.LayerNorm(hidden_channels) 
        
    def forward(self, x, t_in, supports, t_out, w_type):#xtorch.Size([32, 2, 50, 36]) w_type  torch.Size([32, 50, 36])

#         if self.use_cov:
#             w_vec = self.w_embedding(w_type)#torch.Size([32, 50, 36, 4])
#             w_vec = w_vec.permute(0, 3, 1, 2) #torch.Size([32, 4, 50, 36]
#             x = torch.cat([x, w_vec], 1) #torch.Size([32, 6, 50, 36])


        batch_size=x.shape[0]
        node_embedding1 = self.node_embeddings1 #N,H
        node_embeddings=[node_embedding1,self.node_embeddings1]
    

        
        
        node_embeddings=[node_embedding1,self.node_embeddings1]
        if x.shape[1]==self.in_channels:
            w_vec=w_type.unsqueeze(1)
            x = torch.cat([x, w_vec], 1)
            x=self.x_e(x.permute(0,3,2,1)).permute(0,3,2,1) #B,H,N,T
            
            
            filter = self.fc(x.permute(0,3,2,1)) #B,T,N,H
            nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))  #torch.Size([32, 12, 70, 32])B,T,N,H
            nodevec=nodevec.permute(0,2,1,3).reshape(batch_size,self.node_nums,-1) #torch.Size()B,N,T,H
            
            time_emb=sin_cos_time_embed(t_in,self.hidden_channels).permute(0,2,1).unsqueeze(2)#32,64,36
#             node_emb=self.node_embed(self.lap_pos_enc.to(time_emb.dtype)).permute(1,0).unsqueeze(2).unsqueeze(0)
            node_emb=self.lap_pos_enc.to(time_emb.dtype).permute(1,0).unsqueeze(2).unsqueeze(0)
            x=x+time_emb+node_emb
            
        else:
            filter = self.fc(x.permute(0,3,2,1)) #B,T,N,H
            nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))  #torch.Size([32, 12, 70, 32])B,T,N,H
            nodevec=nodevec.permute(0,2,1,3).reshape(batch_size,self.node_nums,-1) #torch.Size()B,N,T,H
        
        
        
        supports1 = torch.eye(self.node_nums) #N,N
        supports1 = supports1.to(x.device)
#         x = torch.einsum("nm,bhmt->bhnt", supports1, x)
        
        supports2 = get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))), supports1)
        
        x =  x.permute(0,2,3,1)+self.norm1(self.time_attn1(x))
        x = x.permute(0,3,1,2)
        x_s = torch.einsum("bnm,bhmt->bhnt", supports2, x) #[32, 32, 70, 12]
        x_s = x_s.permute(0,2,3,1)
        x = x+(self.norm2(x_s)).permute(0,3,1,2)
        for i, layer in enumerate(self.predictlayers): #32, 50, 36, 64
            x = layer(x, supports[i])  # supports[0], supports[1], supports[2]

            if(i<self.num_layers-1):
                x=x.permute(0,3,1,2)
#         x=self.predict_layers(x,supports[0]) #32, 50, 36, 64


#         x=self.prediction_generator1(x) #32, 50, 36, 2
#         x=self.prediction_generator2(x.permute(0,3,1,2)) #32, 2, 50, 12
#         x=self.node_attn(x)


 
        
        
        return x #torch.Size([B, N, T, H])
def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer
class DiffusionEmbedding(nn.Module):

    def __init__(self, num_steps, embedding_dim, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
class FusionBlock(nn.Module):

    def __init__(self, side_dim, channels, diffusion_embedding_dim):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(2*side_dim, channels, 1)



    def forward(self, x, cond_info, diffusion_emb): #torch.Size([32, 64, 50, 36]) B,d_model,N,T
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
        y = x + diffusion_emb  # (B,channel,K,L)

        _, F_dim, _, _, cond_dim = cond_info.shape #32, 2, 50, 36, 65
        
        cond_info=cond_info.permute(0,1,4,2,3)
        
        cond_info = cond_info.reshape(B, F_dim*cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,channel,K*L)
        y = y + cond_info.reshape(B, channel, K * L)
        
        y = y.reshape(B, channel, K , L)
        
        return y

class Decoder(nn.Module): #加门控
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.output_projection1 = Conv1d_with_init(channels, channels, 1)
        self.output_projection2 = Conv1d_with_init(channels, 2, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.x_conv = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.encoder_output_conv = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.conv1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
    def forward(self, x_hidden,encoder_output, B, K, L):  # (B, channel, K*L) => (B, K, L)
        x = x_hidden.reshape(B,-1,K,L)
        x = self.x_conv(x)
        encoder_output=self.encoder_output_conv(encoder_output)
        temp=(x+torch.sigmoid(encoder_output)).contiguous()
        out=(F.relu(temp + self.conv1(encoder_output))).contiguous()
        out=out.reshape(B,-1,K*L)
        out = self.output_projection2(out)
        out = out.reshape(B,-1,K,L)
        
        
        return out
    
class GLU(nn.Module): #加门控
    def __init__(self, channels):
        super(GLU, self).__init__()
        
        
        self.total_inputs_conv = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.encoder_output_conv = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.conv1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        
    def forward(self,total_inputs,encoder_output):  # (B, channel, K*L) => (B, K, L)
        
        total_inputs = self.total_inputs_conv(total_inputs) #B,H,N,T
        
        encoder_output=self.encoder_output_conv(encoder_output)
        
        temp=(total_inputs+torch.sigmoid(encoder_output)).contiguous()
        
        out=(F.relu(temp + self.conv1(encoder_output))).contiguous()
        
        
        return out
    
    

class Model(nn.Module):

    def __init__(self, config,alphas,betas,alpha_bar):
        super(Model, self).__init__()
        
        self.device = config['device']
        self.d_model = config['d_model']
        self.num_nodes = config['node_nums']
        self.num_steps=config["num_timesteps"]
        self.hidden=config['hidden']
        self.input_projection = Conv1d_with_init(2, self.hidden, 1)
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_timesteps"],
            embedding_dim=config["d_model"],
        )
        self.in_len=config["in_len"]
        self.out_len=config["out_len"]
        self.alphas=alphas
        self.betas=betas
        self.alpha_bar=alpha_bar
        self.lap_pos_enc=config['lap_pos_enc']
        
        
        
        self.encoder = Encoder(
            lap_pos_enc=self.lap_pos_enc,
            in_len=self.in_len,
            out_len=self.out_len,
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            dropout=config['dropout'],
            heads=config['heads'],
            support_len=config['support_len'],
        ).to(self.device)
 
        self.prediction_generator1=nn.Linear(self.hidden,config['out_channels'])
        self.prediction_generator2=nn.Linear(self.in_len,self.out_len)
        
        
        
#         self.node_embed = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.d_model)
        self.node_embed=nn.Linear(self.hidden,self.hidden)
        self.Fusion = FusionBlock(
                        side_dim=self.d_model+1,
                        channels=self.d_model,
                        diffusion_embedding_dim=self.d_model )
        self.Decoder_decoder = Decoder(self.d_model)
        self.Linear1=nn.Linear(self.d_model,2)
        self.Linear2=nn.Linear(self.d_model,2)
        self.Linear3=nn.Linear(self.d_model,2)
        self.Linear4=nn.Linear(self.in_len,self.out_len)

        self.oeh=nn.Linear(self.out_len,self.in_len)
        self.forecast=nn.Linear(self.in_len,self.out_len)
        self.total_inputs_embed=nn.Linear(config['in_channels'],self.d_model)
        self.GluLayer=GLU(self.d_model)
        self.NoiseLinear=nn.Linear(self.d_model,config['in_channels'])
        
        self.valueLinear=nn.Linear(self.d_model,config['in_channels'])
        self.value=nn.Linear(self.in_len,self.out_len)
    def __embedding(self, x,  B, F_dim, N, T):
        if x is None:  # for pred_x in validation phase
            return None
        x = x.reshape(B, F_dim, N * T)
        x = self.input_projection(x)

        x = F.relu(x)
        x = x.reshape(B, self.hidden, N, T)
        return x
#(x_history, B, F, N, T, side_info.shape[4])
    
    
    
    def pretrain(self, x_history, t_in, supports, t_out, w_type,x_noisy,input_mask,original_mask,indicating_mask,t): #x_history:([32, 2, 50, 36]),x_noisytorch.Size([32, 2, 50, 12])
        
        B, F, N, T=input_mask.shape
        
        reverse_mask = original_mask * indicating_mask #原数据没缺失，人工缺失
        
        mask_x=x_history*input_mask #与人工缺失+原始缺失的mask矩阵乘后的x
        
        negative_mask = torch.zeros_like(mask_x)
        
        noisy_target = ((1 - input_mask) * x_noisy)
        
        total_inputs = mask_x+noisy_target #torch.Size([B, 2, N, T]) ##※
        
        total_inputs_emb = self.total_inputs_embed(total_inputs.permute(0,3,2,1)).permute(0,3,2,1)#B,H,N,T
        
        reverse_total_input = x_history*reverse_mask + ((1 - reverse_mask) * x_noisy)
        
        time_embedding = sin_cos_time_embed(t_in,self.d_model)  #32,36,64
        
        time_emb=time_embedding.permute(0,2,1).unsqueeze(2)
        
        
        time_embedding = time_embedding.unsqueeze(1).unsqueeze(1).expand(
            -1, F, N, -1, -1)  # (B,F, N, T, emb_time) 32,2,50,36,64
        

#         node_embedding = self.node_embed(torch.arange(self.num_nodes).to(self.device)) #50,64
        
        
        
        node_embedding=self.node_embed(self.lap_pos_enc.to(time_embedding.dtype).to(self.device))
        
        node_emb=node_embedding.permute(1,0).unsqueeze(2).unsqueeze(0)
        
        total_inputs_emb=total_inputs_emb+time_emb+node_emb #B,H,N,T
        
        
        
        node_embedding = (
            node_embedding.unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(B, F, -1,T, -1)
        )  # (B,F, N, T, emb_time) 32,2,50,36,64

        diffusion_emb = self.diffusion_embedding(t) #32,64

        #位置信息：告诉模型哪些需要填补
        side_infomation= time_embedding+node_embedding
        
        side_info = torch.cat([side_infomation, input_mask.unsqueeze(4)], dim=4) #32, 2, 50, 36, 65
        
        reverse_side_info = torch.cat([side_infomation, reverse_mask.unsqueeze(4)], dim=4)
        
        negative_side_info = torch.cat([side_infomation, negative_mask.unsqueeze(4)], dim=4)
        

        
        
        
        
        #encoder
        encoder_output=self.encoder(x_history,t_in,supports,t_out,w_type) # #torch.Size([B, N, T, H])
        
        encoder_output_hidden1=encoder_output.permute(0,3,1,2) #torch.Size([32, 64, 50, 36])B,H,N,T
        
        
        
        
        
        
        #解码器嵌入：total_inputs，reverse_total_input

        
        
        
        
        #decoder,插补
        forward_noise_hidden = self.Fusion(total_inputs_emb, side_info, diffusion_emb)#B,H,N,T

        

        
        """
        GLU
        """
        
        Glu_output=self.GluLayer(forward_noise_hidden,encoder_output_hidden1) #B,H,N,T
        

        predicted_noise=self.NoiseLinear(Glu_output.permute(0,2,3,1)) #torch.Size([32, 70, 36, 2])
        
        predicted_noise = predicted_noise.permute(0,3,1,2) #32,2,70,36
         
        """
        GLU
        """
        
        
        
        
        
        return predicted_noise

    
    
    def trains(self, x_history, t_in, supports, t_out, w_type): #x_history:([32, 2, 50, 36]),x_noisytorch.Size([32, 2, 50, 12])
        
        

        encoder_output = self.encoder(x_history,t_in,supports,t_out,w_type) #torch.Size([32, 2, 50, 36])
        
        
        encoder_output=self.prediction_generator1(encoder_output) #32, 50, 36, 2
        encoder_output=self.prediction_generator2(encoder_output.permute(0,3,1,2)) #32, 2, 50, 12
        
        
        return encoder_output
    
    
    
    
    

class DST_DAMP:
    def __init__(self, config):
        self.device = config['device']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.num_timesteps = config['num_timesteps']

        
        self.betas = self.get_beta_schedule(config['beta_schedule'], 
                                          config['num_timesteps'],
                                          config['beta_start'],
                                          config['beta_end'])
        
        self.alphas = 1. - self.betas
        
        
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alpha_bar=self.alphas_cumprod
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sigma2 = self.betas
        
        
        self.model=Model(config,self.alphas,self.betas,self.alpha_bar).to(self.device)
        
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=config['lr'], 
                                  weight_decay=config['decay'])
    
    def get_beta_schedule(self, schedule_name, num_timesteps, beta_start=0.0001, beta_end=0.02):
        if schedule_name == 'linear':
            return torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_name == 'cosine':
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos((x / num_timesteps + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        elif schedule_name == 'quad':
            
            return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_name}")

    
    def q_xt_x0(self, y, t, noise=None):
        """
        Sample from  q(x_t|x_0) ~ N(x_t; \sqrt\bar\alpha_t * x_0, (1 - \bar\alpha_t)I)
        """
        if noise is None:
            noise = torch.randn_like(y)
        

        mean = gather(self.alpha_bar, t).to(self.device) ** 0.5 * y
        var = 1 - gather(self.alpha_bar, t)
        return mean + noise * (var.to(self.device) ** 0.5)

    def val_step(self, batch):
        x, t_in, t_out, w_type ,y,supports,input_mask,original_mask,indicating_mask,y_original_mask= batch
        x = x.to(self.device)
        t_in = t_in.to(self.device)
        t_out = t_out.to(self.device)
        w_type = w_type.to(self.device)
        y = y.to(self.device)
        input_mask = input_mask.to(self.device)
        original_mask = original_mask.to(self.device)
        indicating_mask = indicating_mask.to(self.device)
        y_original_mask = y_original_mask.to(self.device)
        reverse_cond_mask = (original_mask * indicating_mask).to(self.device)

        B, _, N, T = x.shape

        with torch.no_grad():  # Disable gradient calculation for validation
            # Sample random timesteps
            t = torch.randint(0, self.num_timesteps, (x.shape[0],)).long().to(self.device)
            noise = torch.randn_like(x)
            x_noisy = self.q_xt_x0(x, t, noise)

            predicted_noise= self.model.pretrain(
                x, t_in, supports, t_out, w_type, x_noisy, input_mask, original_mask, indicating_mask, t
            )

            target_mask = original_mask - input_mask
        
            residual = (x - predicted_noise) * target_mask

            num_eval = target_mask.sum()

            diff_loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1) 

#             predict_hidden = forward_noise_hidden.reshape(B, -1, N, T) * (1 - input_mask) + \
#                             reverse_noise_hidden.reshape(B, -1, N, T) * (1 - reverse_cond_mask) 

#             loss_cons = self.calc_intra_cons_loss(
#                 negative_hidden.reshape(B, -1), predict_hidden.reshape(B, -1)
#             )


            # You might want to calculate additional metrics here
            # For example:

        total_loss = diff_loss.item()

        return total_loss
   
    def pretrain_step(self, batch):
        x, t_in, t_out, w_type ,y,supports,input_mask,original_mask,indicating_mask,y_original_mask= batch
        x = x.to(self.device)
        t_in = t_in.to(self.device)
        t_out = t_out.to(self.device)
        w_type = w_type.to(self.device)
        y=y.to(self.device)
        input_mask=input_mask.to(self.device)
        original_mask=original_mask.to(self.device)
        indicating_mask=indicating_mask.to(self.device)
        y_original_mask=y_original_mask.to(self.device)
        reverse_cond_mask = (
            original_mask * indicating_mask
        ).to(self.device)
        
        B,_,N,T=x.shape
        self.optimizer.zero_grad()
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (x.shape[0],)).long().to(self.device)
        noise = torch.randn_like(x)
        x_noisy = self.q_xt_x0(x, t, noise)
        
        predicted_noise= self.model.pretrain(x, t_in, supports, t_out, w_type,x_noisy,input_mask,original_mask,indicating_mask,t)
        
        target_mask = original_mask - input_mask

        residual = (x - predicted_noise) * target_mask
        
        
        num_eval = target_mask.sum()
        

        
        loss_noise = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        

#         predict_hidden = forward_noise_hidden.reshape(B, -1, N, T) * (
#             1 - input_mask
#         ) + reverse_noise_hidden.reshape(B, -1, N, T) * (1 - reverse_cond_mask) + encoder_output_hidden.reshape(B, -1, N, T) * (1 - input_mask)
#         predict_hidden = forward_noise_hidden.reshape(B, -1, N, T) * (
#             1 - input_mask
#         ) + reverse_noise_hidden.reshape(B, -1, N, T) * (1 - reverse_cond_mask) 
        
        
        
        
#         loss_cons = self.calc_intra_cons_loss(
#             negative_hidden.reshape(B, -1), predict_hidden.reshape(B, -1)
#         )
        loss = loss_noise 
#         loss = F.mse_loss(noise, predicted_noise)
        
#         loss = self.p_losses(x, t, t_in, supports, t_out, w_type,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
        self.optimizer.step()
        return loss.item()
    
    
    
    
    
    def train_step(self,  batch):
        x, t_in, t_out, w_type ,y,supports,original_mask= batch
        x = x.to(self.device)
        t_in = t_in.to(self.device)
        t_out = t_out.to(self.device)
        w_type = w_type.to(self.device)
        y =  y.to(self.device)
        original_mask=original_mask.to(self.device)
        B,_,N,T=x.shape
        self.model.train()
        self.optimizer.zero_grad()
        
        # Sample random timesteps
#         for name, param in self.model.named_parameters():
#             if torch.isnan(param).any():
#                 print(f"NaN detected in parameter: {name}")
        predicted = self.model.trains(x, t_in, supports, t_out, w_type)

    
        residual = (y - predicted) * original_mask
        
        num_eval = original_mask.sum()
    
#         loss=masked_rmse(predicted, y,0.0)
        loss=(residual**2).sum() / (num_eval if num_eval > 0 else 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
        self.optimizer.step()
        
#         loss.backward()
#         self.optimizer.step()
        
        return loss.item()
#         loss=masked_rmse(predicted, y)
#         loss = F.mse_loss(noise, predicted_noise)
        
#         loss = self.p_losses(x, t, t_in, supports, t_out, w_type,y)
#         loss.backward()
#         self.optimizer.step()
    
    
    def test_step(self,  batch): 
        x, t_in, t_out, w_type ,supports= batch
        x = x.to(self.device)
        t_in = t_in.to(self.device)
        t_out = t_out.to(self.device)
        w_type = w_type.to(self.device)
        
        B,_,N,T=x.shape
        self.model.eval()
        
        # Sample random timesteps
#         for name, param in self.model.named_parameters():
#             if torch.isnan(param).any():
#                 print(f"NaN detected in parameter: {name}")
        predicted = self.model.trains(x, t_in, supports, t_out, w_type)
        
#         loss.backward()
#         self.optimizer.step()
        
        return predicted
#         loss=masked_rmse(predicted, y)
#         loss = F.mse_loss(noise, predicted_noise)
        
#         loss = self.p_losses(x, t, t_in, supports, t_out, w_type,y)
#         loss.backward()
#         self.optimizer.step()
    
    
    
    



 


