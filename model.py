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


                result = result+(self.beta*torch.mul(torch.unsqueeze(Attention[i][:,j],dim=1),semantic_embeddings[j][2]))
            result+=semantic_embeddings[j][3]
            result_embeddings.append(result)
        final_embedding = torch.cat((result_embeddings),dim=1)
        return self.predict(final_embedding)
