import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HANConv, GATv2Conv
from typing import Dict, List, Union

from torch_geometric.utils import to_dense_adj,dense_to_sparse, softmax,subgraph
from torch_geometric.utils.dropout import dropout_edge, dropout_node
import random
#from layers import GraphConvolution
import warnings
warnings.filterwarnings("ignore")


class GCN(torch.nn.Module):
	def __init__(self, in_dim, hidden, out_dim, dropout, layer_s):
		super(GCN, self).__init__()
		self.layers = []
		#self.enc = GATConv(in_dim, hidden)
		self.enc = nn.Linear(in_dim, hidden)
		for i in range(layer_s):
			self.layers.append(GCNConv(hidden,hidden))

		self.dec = nn.Linear(hidden,out_dim)
		#self.dec = GATConv(hidden,out_dim)

		self.layers = torch.nn.ModuleList(self.layers)
		self.dropout = dropout
		
	def forward(self,x,g):
		x = F.dropout(x,self.dropout,training=self.training)
		#print(f'g,{g}')

		#g, _ = dropout_edge(g,p=self.dropout,force_undirected=True,training=self.training)
		#g, _, _ = dropout_node(g,p=self.dropout,training=self.training)
		#print(f'drop_g,{g}')
		#x0 = self.enc(x)
		#x = self.enc(x)
		x = F.leaky_relu(self.enc(x),0.1)
		#x = F.leaky_relu(self.enc(x,g),0.2)
		for i,conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,g),0.1)
			#x = conv(x,g)+x0
			
		x = self.dec(x)
		return x
	@torch.no_grad()
	def getembedding(self,x,g):
		# x = F.dropout(x,self.dropout,training=self.training)
		x = F.leaky_relu(self.enc(x,g),0.2)
		for i,conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,g),0.2)
		return x

class GAT(torch.nn.Module):
	def __init__(self, in_dim, hidden, out_dim, dropout, layer_s):
		super(GAT, self).__init__()
		self.layers = []
		#self.enc = GATv2Conv(in_dim, out_dim)
		self.enc = nn.Linear(in_dim, hidden)
		for i in range(layer_s):
			self.layers.append(GATv2Conv(hidden,hidden))

		#self.dec = nn.Linear(hidden,out_dim)
		#self.dec = GATv2Conv(hidden,out_dim)

		self.layers = torch.nn.ModuleList(self.layers)
		self.dropout = dropout
		
	def forward(self,x,g):
		x = F.dropout(x,self.dropout,training=self.training)
		#print(f'g,{g}')

		#g, _ = dropout_edge(g,p=0.9,force_undirected=True,training=self.training)
		#g, _, _ = dropout_node(g,p=self.dropout,training=self.training)
		#print(f'drop_g,{g}')
		#x0 = self.enc(x)
		#x = self.enc(x)
		x = F.leaky_relu(self.enc(x),0.1)
		#x = F.leaky_relu(self.enc(x,g),0.2)
		for i,conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,g),0.1)
			#x = conv(x,g)+x0
			
		#x = self.dec(x)
		return x
	@torch.no_grad()
	def getembedding(self,x,g):
		# x = F.dropout(x,self.dropout,training=self.training)
		x = F.leaky_relu(self.enc(x,g),0.2)
		for i,conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,g),0.2)
		return x

class GraphSAGE(torch.nn.Module):
	def __init__(self, in_dim, hidden, out_dim, dropout, layer_s):
		super(GraphSAGE, self).__init__()
		self.layers = []
		#self.enc = GATConv(in_dim, hidden)
		self.enc = nn.Linear(in_dim, hidden)
		for i in range(layer_s):
			self.layers.append(SAGEConv(hidden,hidden))

		self.dec = nn.Linear(hidden,out_dim)
		#self.dec = GATConv(hidden,out_dim)

		self.layers = torch.nn.ModuleList(self.layers)
		self.dropout = dropout
		
	def forward(self,x,g):
		x = F.dropout(x,self.dropout,training=self.training)
		#print(f'g,{g}')

		#g, _ = dropout_edge(g,p=0.2,force_undirected=True,training=self.training)
		#g, _, _ = dropout_node(g,p=self.dropout,training=self.training)
		#print(f'drop_g,{g}')
		#x0 = self.enc(x)
		#x = self.enc(x)
		x = F.leaky_relu(self.enc(x),0.1)
		#x = F.leaky_relu(self.enc(x,g),0.2)
		for i,conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,g),0.1)
			#x = conv(x,g)+x0
			
		#x = self.dec(x)
		return x

class basic_HAN(nn.Module):
	def __init__(self, in_dim, hidden, out_dim, atten_dim, metadata, num_heads=2):
		super(basic_HAN, self).__init__()

		# Metadata for different meta-paths
		self.metadata = metadata
		self.num_metapaths = len(metadata[1])  # Number of meta-paths
		self.num_heads = num_heads

		# Attention layers for each meta-path (node-level attention)
		self.NAtts = nn.ModuleList([self.NAtt(in_dim, hidden, num_heads) for _ in range(self.num_metapaths)])

		# Semantic-level attention layer
		self.SAtts = self.SAtt(hidden,atten_dim)

		self.lin = nn.Linear(hidden, out_dim)

	class NAtt(nn.Module):
		def __init__(self, in_dim, out_dim, num_heads=2):
			super().__init__()
			self.gat_list = nn.ModuleList()

			for i in range(num_heads):
				#self.gat_list.append(GraphSAGE(in_dim=in_dim, hidden=out_dim, out_dim=int(out_dim/num_heads), dropout=0, layer_s=1))
				self.gat_list.append(GCN(in_dim=in_dim, hidden=out_dim, out_dim=int(out_dim/num_heads), dropout=0, layer_s=1))

		def forward(self, x, edge_index):
			h_prime = torch.concat([self.gat_list[i](x,edge_index) for i in range(len(self.gat_list))],dim=1)

			return h_prime

	class SAtt(nn.Module):
		def __init__(self, in_dim,atten_dim):
			super().__init__()
			self.lin = nn.Linear(in_dim,atten_dim)
			self.q_vector = nn.Linear(atten_dim,1,bias=False)

		def forward(self, metapaths):
			weights = []
			for z in metapaths:
				h = torch.tanh(self.lin(z))
				beta = self.q_vector(h).mean(0)
				weights.append(beta)


			# Normalize weights(B)
			weights = torch.stack(weights)
			weights = F.softmax(weights, dim=0)

			# Fuse semantic-specific embeddings
			Z = sum(w * z for w, z in zip(weights, metapaths))
			return Z, weights
	def forward(self, x_dict, edge_index_dict):
		z_metapaths = []
		alpha_list = []  # To store node-level attention weights

		for i, edge_type in enumerate(self.metadata[1]):
			edge_index = edge_index_dict[edge_type]
			x = x_dict[self.metadata[0][0]]  # Assuming a single node type

			# Step 4-11: Meta-path attention for each head and neighbor aggregation
			z_metapath = self.NAtts[i](x, edge_index)
			z_metapaths.append(z_metapath)

		# Step 13-14: Semantic-level attention to fuse embeddings across meta-paths
		Z, beta = self.SAtts(z_metapaths)
		# Final prediction layer
		out = self.lin(Z)

		return F.log_softmax(out, dim=-1), alpha_list, beta

class AttentionH(nn.Module):
	def __init__(self, feat_dim, atten_dim,num_gcn):
		super(AttentionH, self).__init__()
		self.attention_list = nn.ModuleList()
		self.num_gcn = num_gcn
		self.att = torch.nn.Linear(feat_dim,atten_dim)
		for i in range(num_gcn):
			self.attention_list.append(torch.nn.Linear(feat_dim,atten_dim))
		self.agg = torch.nn.Linear(num_gcn*atten_dim,num_gcn)
	def forward(self,embed):
		gcn_attention = []
		#mean_embed = torch.mean(embed,dim=0)
		for i in range(self.num_gcn):
			#gcn_attention.append(self.attention_list[i].forward(embed[i]))
			#print(f'mean:{torch.mean(embed[i],dim=0).unsqueeze(0).shape}')
			gcn_attention.append(self.attention_list[i].forward(embed[i]))
			#gcn_attention.append(self.attention_list[i].forward(torch.mean(embed[i],dim=0).unsqueeze(0)))
		#print(f'gcn:{gcn_attention[0].shape}')
		#attention = torch.softmax(self.agg(torch.cat(tuple(gcn_attention),dim=1)),dim=1)

		# ------ min max way ---------------
		attention = self.agg(torch.cat(tuple(gcn_attention),dim=1)) #[node,num_gcn]
		attention = attention-torch.mean(attention,dim=1).unsqueeze(1)
		atten_min = torch.min(attention,dim=1).values.unsqueeze(1)
		atten_max = torch.max(attention,dim=1).values.unsqueeze(1)
		attention = (attention-atten_min)/(atten_max-atten_min)
		#attention = attention-torch.mean(attention,dim=1).unsqueeze(1)
		final_embed = (attention[:,0].unsqueeze(1))*embed[0]
		#print('att',torch.max(attention))
		#print('attmin',torch.min(torch.abs(attention)))
		#print(torch.min(attention))
		#print(f'final_embed: {final_embed.shape}')
		for i in range(1,self.num_gcn):
			final_embed += (attention[:,i].unsqueeze(1)+(1/self.num_gcn))*embed[i]

		# --------- transformer way ---------------------
		# attention = torch.softmax(self.agg(torch.cat(tuple(gcn_attention),dim=1)),dim=1)
		# final_embed = (attention[:,0].unsqueeze(1))*embed[0]
		# for i in range(1,self.num_gcn):
		# 	final_embed += (attention[:,i].unsqueeze(1))*embed[i]

		# --------- residual way ---------------------
		# attention = torch.softmax(self.agg(torch.cat(tuple(gcn_attention),dim=1)),dim=1)
		# final_embed = (attention[:,0].unsqueeze(1)+1)*embed[0]
		# for i in range(1,self.num_gcn):
		# 	final_embed += (attention[:,i].unsqueeze(1)+1)*embed[i]



		return final_embed, attention

class Attention(nn.Module):
	def __init__(self, in_dim, hidden, out_dim):
		super(Attention, self).__init__()

		self.hq = torch.nn.Linear(in_dim, hidden)
		self.hk = torch.nn.Linear(hidden, 1)
		self.mq = torch.nn.Linear(hidden, hidden)
		self.mk = torch.nn.Linear(hidden,1)
		self.out = torch.nn.Linear(hidden, out_dim)

	def forward(self, feat):
		print(feat.shape) #how many models used
		final_list = []
		for i in range(feat.shape[0]):
			x = feat[i]
			x = F.tanh(self.hq(x))

			scores = self.hk(x)
			weights = F.softmax(scores, dim=0)
			#[2,13334,128]
			#print('weights',weights)
			final_embedding = (weights * x).sum(dim=0)
			final_list.append(final_embedding)
		stacked = torch.stack(final_list)
		x = stacked
		#print(x.shape)
		x = F.tanh(self.mq(x))

		scores = self.mk(x)
		weights = F.softmax(scores, dim=0)
		#[2,13334,128]
		#print('weights',weights)
		final_embedding = (weights * x).sum(dim=0)
		final_embedding = self.out(final_embedding)
		return final_embedding
	@torch.no_grad()
	def getembedding(self,x):
		x = F.tanh(self.a(x))
		scores = self.b(x)
		weights = F.softmax(scores, dim=0)
		#[2,13334,128]
		#print('weights',weights)
		final_embedding = (weights * x).sum(dim=0)
		return final_embedding

class MultiGCN(nn.Module):
	def __init__(self, in_dim, hidden, out_dim, attention_dim,num_gcn, dropout, layer_s, num_path):
		super(MultiGCN, self).__init__()
		self.gcn_list = nn.ModuleList()
		self.num_path = num_path
		self.num_gcn = num_gcn

		# Initialize GCNs for each metapath
		for i in range(num_path):
			self.gcn_list.append(nn.ModuleList([GCN_embed(in_dim, hidden, hidden, dropout, layer_s) for _ in range(num_gcn)]))
			#self.gcn_list.append(nn.ModuleList([GraphSAGE(in_dim, hidden, hidden, dropout,layer_s) for _ in range(num_gcn)]))
			#self.gcn_list.append(nn.ModuleList([GAT(in_dim, hidden, hidden, dropout,layer_s) for _ in range(num_gcn)]))
		self.attention_list = nn.ModuleList([AttentionH(feat_dim=hidden, atten_dim=attention_dim, num_gcn=self.num_gcn) for _ in range(num_path)])
		#self.attention_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden, num_heads=1, batch_first=True) for _ in range(num_path)]) #transformer
		#self.dec = nn.Linear(hidden, out_dim)
		self.dec_list = nn.ModuleList()
			
		for i in range(num_path):
			self.dec_list.append(nn.Linear(hidden,out_dim))


	def forward(self, data_list):
	# Store predictions for each GCN model

		predictions = []

		for idx, data in enumerate(data_list):
			#prediction_mean = []
			for model in self.gcn_list[idx]:
				#prediction_mean.append(model(data.x, data.edge_index))
				predictions.append(model(data.x, data.edge_index))
			#predictions.append(torch.sum(torch.stack(prediction_mean),dim=0))
		#print(f'pre:{predictions[0].shape}')



		# Stack predictions and aggregate for final prediction

		predictions_total = torch.stack(predictions) #[6,13334,2]
		#print(f'pred:{predictions_total.shape}')
		# unique_predictions, counts = torch.unique(predictions_total, dim=0, return_counts=True)
		# print('c',counts)
		node_std = []
		difference = []
		cov = []
		mean = []
		#--------------------------------------------- WMulti(change hidden to out, and add dec)
		# # Loop over each node
		# for i in range(self.num_path):
		# 	start = i*self.num_gcn
		# 	end = start+self.num_gcn
		# 	node_pred = predictions_total[start:end, :, :]  #[4054,4]
		# 	node_mean = torch.mean(node_pred, dim=0, keepdim=True)  #[4054, 4]
		# 	mean.append(node_mean)
		# stack_mean = torch.stack(mean) #[metapath,1,node,class]#
		# #print(stack_mean.shape)
		# # combined_means = torch
		# combined_means = stack_mean.view(-1, stack_mean.shape[-1])
		# #print(combined_means.shape)
		# combined_cov = torch.cov(combined_means.T)
		# cov_mean = combined_cov

		# final_prediction = torch.sum(predictions_total,dim=0) #[4057,4]
		# out=0
		# print(combined_cov.shape)
		# ----------------------------------------------------------

		path_predictions_total = []
		pool_list = []
		
        # # #---------------------------------------- HGEN/transformer(hidden,no dec)
		for idx, data in enumerate(data_list):
			path_predictions = []

		# Collect predictions from 3 GCN models for the current metapath
			for model in self.gcn_list[idx]:
				path_predictions.append(model(data.x, data.edge_index))

			# Stack predictions for the current metapath
			#path_predictions = torch.stack(path_predictions, dim=1)  #[num_nodes, num_gcn, out_dim]
			#print(f'input:{path_predictions[0].shape}')

			# Use attention to transform predictions for the current path
			att_layer = self.attention_list[idx]
			#attn_output, weight = att_layer(path_predictions, path_predictions, path_predictions,need_weights=True) #transformer
			attn_output, weight = att_layer(path_predictions)
			#print(f'attention output :{attn_output.shape}')
			#print(f'attention w :{weight.shape}') #[num_nodes, num_gcn, num_gcn]
			#print(f'attention w :{torch.mean(weight,dim=0)}')
			#print(f'attention std :{torch.std(weight,dim=0)}')
			#transformed_path_output = attn_output.sum(dim=1)  #[num_nodes, out_dim]
			transformed_path_output = attn_output
			attn_pool = torch.mean(transformed_path_output,dim=0)
			pool_list.append(attn_pool)
			# Store the transformed output
			path_predictions_total.append(transformed_path_output)
		cov = torch.stack(pool_list)
		cov_mean = torch.matmul(cov, cov.T)
		#print(cov_mean.shape)



		final_predictions = []
		out = torch.sum(torch.stack(path_predictions_total),dim=0)
		#final_predictions = self.dec(torch.sum(torch.stack(path_predictions_total),dim=0)) #[num_nodes, out_dim]
		for i in range(self.num_path):
			single_prediction = self.dec_list[i](path_predictions_total[i])
			final_predictions.append(single_prediction) #[num_nodes, out_dim]
		final_prediction = torch.sum(torch.stack(final_predictions),dim=0)
		# #----------------------------------------------------------------------------
		#print(final_predictions.shape)
		# for i in range(predictions_total.shape[1]):  #4054

		# 	node_pred = predictions_total[:, i, :]  #[9, 4]
		# 	node_mean = torch.mean(node_pred, dim=0, keepdim=True)  #[1, 4]
		# 	centered_pred = node_pred - node_mean #[9, 4]
		# 	#node_cov = torch.cov(centered_pred)
		# 	#print(node_cov.shape)
		# 	#cov.append(node_cov)

		# 	kl = []
		# 	for j in range(predictions_total.shape[0]):  #9 model
		# 		diff = torch.abs(F.log_softmax(node_mean, dim=-1) - F.softmax(node_pred[j].unsqueeze(0), dim=-1))
		# 		diff = diff.mean()
		# 		difference.append(diff)
		# 		kl_div = F.kl_div(
		# 		F.log_softmax(node_mean, dim=-1),
		# 		F.softmax(node_pred[j].unsqueeze(0), dim=-1),
		# 		reduction='sum'
		# 		)
		# 		kl.append(kl_div)
		# 	kl = torch.tensor(kl)  #[9]
		# 	node_std.append(kl.std())  # Std node

		# std_mean = (torch.stack(node_std)).mean()

		# #print(difference)
		# #mean_diff = difference.mean()

		# print(f'final:{final_predictions[0]}')
		# p = torch.softmax(final_predictions, dim=-1)  # Shape: [N, 4]

		# # Step 2: Calculate entropy
		# epsilon = 1e-10  # Small value to avoid log(0)
		# entropy = (-torch.sum(p * torch.log(p + epsilon), dim=-1)).mean()  # Shape: [N]

		#print(final_predictions)
		#print(final_predictions.shape)
		#return final_predictions, cov_mean
		#return F.log_softmax(final_predictions[1],dim=1), cov_mean
		return F.log_softmax(final_prediction,dim=1), cov_mean, out





	# test single gcn
	@torch.no_grad()
	def forward_single_gcn(self, data_list):
		predictions = []

		acc_list = []
		auc_lsit = []
		for idx, data in enumerate(data_list):
			#prediction_mean = []
			for model in self.gcn_list[idx]:
				#prediction_mean.append(model(data.x, data.edge_index))
				model.eval()
				predictions.append(self.dec_list[idx](model(data.x, data.edge_index)))

		return predictions

	# @torch.no_grad()
	# def forward_predict(self,data_list):
	# 	predictions = []

	# 	for idx, data in enumerate(data_list):
	# 		for model in self.gcn_list[idx]:
	# 			predictions.append(model(data.x, data.edge_index))


	# 	predictions_total = torch.stack(predictions) #[6,13334,2]

	# 	node_std = []
	# 	difference = []
	# 	cov = []
	# 	mean = []

	# 	path_predictions_total = []
	# 	pool_list = []
		
    #     # # #---------------------------------------- HGEN/transformer(hidden,no dec)
	# 	for idx, data in enumerate(data_list):
	# 		path_predictions = []

	# 	# Collect predictions from 3 GCN models for the current metapath
	# 		for model in self.gcn_list[idx]:
	# 			path_predictions.append(model(data.x, data.edge_index))
	# 		att_layer = self.attention_list[idx]
	# 		#attn_output, weight = att_layer(path_predictions, path_predictions, path_predictions,need_weights=True) #transformer
	# 		attn_output, weight = att_layer(path_predictions)

	# 		transformed_path_output = attn_output
	# 		attn_pool = torch.mean(transformed_path_output,dim=0)
	# 		pool_list.append(attn_pool)
	# 		path_predictions_total.append(transformed_path_output)
	# 	cov = torch.stack(pool_list)
	# 	cov_mean = torch.matmul(cov, cov.T)



	# 	final_predictions = []
	# 	out = torch.sum(torch.stack(path_predictions_total),dim=0)
	# 	for i in range(self.num_path):
	# 		single_prediction = self.dec_list[i](path_predictions_total[i])
	# 		final_predictions.append(single_prediction) #[num_nodes, out_dim]
	# 	final_prediction = torch.sum(torch.stack(final_predictions),dim=0)

	# 	return final_prediction

class HAN(nn.Module):
	def __init__(self, in_channels: Union[int, Dict[str, int]],
		out_channels: int,author_data=None, hidden_channels=128, heads=1):
		super().__init__()
		self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
		            dropout=0, metadata=author_data)
		self.lin = nn.Linear(hidden_channels, out_channels)

	def forward(self, x_dict, edge_index_dict):
		out1 = self.han_conv(x_dict, edge_index_dict)
		#out = self.lin(out['author'])
		out = self.lin(out1['movie']) #change with metadata
		return F.log_softmax(out,dim=1),out1

class GCN_embed(torch.nn.Module):
	def __init__(self, in_dim, hidden, out_dim, dropout, layer_s):
		super(GCN_embed, self).__init__()
		self.layers = []
		#self.enc = GATConv(in_dim, hidden)
		self.enc = nn.Linear(in_dim, hidden)
		for i in range(layer_s):
			self.layers.append(GCNConv(hidden,hidden))

		self.dec = nn.Linear(hidden,out_dim)
		#self.dec = GATConv(hidden,out_dim)

		self.layers = torch.nn.ModuleList(self.layers)
		self.dropout = dropout
		
	def forward(self,x,g):
		x = F.dropout(x,self.dropout,training=self.training)
		#print(f'g,{g}')

		#g, _ = dropout_edge(g,p=self.dropout,force_undirected=True,training=self.training)
		#g, _, _ = dropout_node(g,p=self.dropout,training=self.training)
		#print(f'drop_g,{g}')
		#x0 = self.enc(x)
		#x = self.enc(x)
		x = F.leaky_relu(self.enc(x),0.1)
		#x = F.leaky_relu(self.enc(x,g),0.2)
		for i,conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,g),0.1)
			#x = conv(x,g)+x0
			
		#x = self.dec(x)
		return x
	@torch.no_grad()
	def getembedding(self,x,g):
		# x = F.dropout(x,self.dropout,training=self.training)
		x = F.leaky_relu(self.enc(x,g),0.2)
		for i,conv in enumerate(self.layers):
			x = F.leaky_relu(conv(x,g),0.2)
		return x

class HesGNN_agg(nn.Module):
	def __init__(self,in_dim,hid_dim,dropout,layer_s):
		super(HesGNN_agg,self).__init__()
		self.enc = nn.Linear(in_dim, hid_dim)
		#self.enc = SAGEConv(in_dim,hid_dim)
		self.layers = []
		for i in range(layer_s):
			self.layers.append(SAGEConv(hid_dim,hid_dim))
		self.layers = nn.ModuleList(self.layers)
		self.dropout = dropout
	def forward(self,g,x):
		x = F.dropout(x, self.dropout,training=self.training)
		#g, _ = dropout_edge(g,p=0.2,force_undirected=True,training=self.training)
		#x = self.enc(x,g)
		x = self.enc(x)
		for i,conv in enumerate(self.layers):
			x = conv(x,g)
		return x
class SeHGNN(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,num_heads,dropout,layer_s):
        super(SeHGNN,self).__init__()
        self.in_channels = in_dim
        self.hid_channels = hid_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        # self.labels = labels    
        self.Q2 = nn.Linear(hid_dim,hid_dim,bias=True)
        self.K2 = nn.Linear(hid_dim,hid_dim,bias=True)
        self.V2 = nn.Linear(hid_dim,hid_dim,bias=True)
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.predict = nn.Linear(hid_dim*self.num_heads,out_dim)
        models = []
        
        for i in range(self.num_heads):
            gat = HesGNN_agg(in_dim, hid_dim,dropout,layer_s).to("cuda:0")
            #gat = HesGNN_agg(in_dim, hid_dim,dropout,layer_s).to("cuda:0")
            #gat = GraphSAGE(in_dim, hid_dim, out_dim, dropout=0, layer_s=2).to("cuda:0")
            #gat.load_state_dict(torch.load(args.dataset+"_attention_single"+str(i)+"seed"+str(args.seed)+".pth"))
            models.append(gat)
        self.SeHGNN_model = models
        
    def forward(self, adj_list, h):
        #result_list = []
        #for i in range(self.num_heads):
        #   result_list.append(self.GCN(h,adj_list[i].to_sparse()))
        '''attn_out = self.attentionlayer.embed(adj_list,h)
        result_list = attn_out
        semantic_embeddings = torch.stack(
            result_list, dim=1
        )'''
        semantic_embeddings = []
        for i in range(self.num_heads):
            z = self.SeHGNN_model[i].forward(adj_list[i],h)
            #z = self.SeHGNN_model[i].double().forward(adj_list[i],h) #company
            qh = self.Q2(z)
            kh = self.K2(z)
            vh = self.V2(z)
            semantic_embeddings.append((qh,kh,vh,z))
        result_embeddings = []
        Attention = []
        #print(torch.mul(semantic_embeddings[0][0],semantic_embeddings[0][1]).shape)
        for i in range(self.num_heads):
            attention = []
            for j in range(self.num_heads):
                attention.append((torch.sum(torch.mul(semantic_embeddings[i][0],semantic_embeddings[j][1]),axis=1)))

            #print(torch.stack(attention,dim=1).shape)
            attention = torch.softmax(torch.stack(attention,dim=1),axis=1)
            Attention.append(attention)
        result_embeddings = []
        for i in range(self.num_heads):
            result = torch.zeros(semantic_embeddings[j][3].shape).to("cuda:0")
            for j in range(self.num_heads):
                result = result+(self.beta*torch.mul(torch.unsqueeze(Attention[i][:,j],dim=1),semantic_embeddings[j][2]))
            result+=semantic_embeddings[j][3]
            result_embeddings.append(result)
        final_embedding = torch.cat((result_embeddings),dim=1)
        return self.predict(final_embedding)