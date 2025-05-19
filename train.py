import numpy as np
import torch
from dataset import load_DBLP, load_company, load_IMDB, load_ACM, load_Urban
import argparse
from model import GCN,GAT,GraphSAGE, MultiGCN, HAN,basic_HAN,SeHGNN
from torch_geometric.utils import to_dense_adj,dense_to_sparse,degree
#from torch_geometric.loader import NeighborLoader
from torch_scatter import scatter
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import ray
import seaborn as sns
import warnings
from ray import tune,train
from functools import partial
import torch.nn.functional as F
import os
import copy

warnings.filterwarnings("ignore")

def delete_file(filepath):
	"""
	Delete a file if it exists.

	Args:
	filepath (str): Path to the file to be deleted.

	Returns:
	bool: True if the file was deleted, False if the file did not exist.
	"""
	if os.path.exists(filepath):
		os.remove(filepath)  # Delete the file
		print(f"File '{filepath}' has been deleted.")
		return True
	else:
		print(f"File '{filepath}' does not exist.")
		return False

def run_Multi(args,count):

	if args.dataset == 'Company':
		data,data_COC, data_CPC, train_mask, val_mask, test_mask = load_company(seed=args.seed, dataname='',path='')
		in_dim = data_COC.x.shape[1]
		num_class = data_COC.y.unique().shape[0] 
		data = data.to(args.device)
		data_COC = data_COC.to(args.device)
		data_CPC = data_CPC.to(args.device)
		graphs = [data_COC,data_CPC]
	elif args.dataset == 'DBLP':

		data, data_APA, data_APCPA, data_APTPA, train_mask, val_mask, test_mask = load_DBLP(seed=args.seed, dataname='DBLP',path='')
		in_dim = data_APA.x.shape[1]
		num_class = data_APA.y.unique().shape[0]
		data_APA = data_APA.to(args.device)
		data_APCPA = data_APCPA.to(args.device)
		data_APTPA = data_APTPA.to(args.device)
		graphs = [data_APA,data_APTPA,data_APCPA]
	elif args.dataset == 'IMDB':

		data,data_MAM, data_MDM,data_AMA, data_DMD, train_mask, val_mask, test_mask = load_IMDB(seed=args.seed, path='')
		in_dim = data_MAM.x.shape[1]
		num_class = data_MAM.y.unique().shape[0] 
		data = data.to(args.device)
		data_MAM = data_MAM.to(args.device)
		data_MDM = data_MDM.to(args.device)
		data_AMA = data_AMA.to(args.device)
		data_DMD = data_DMD.to(args.device)
		graphs = [data_MAM,data_MDM,data_AMA,data_DMD]
	elif args.dataset == 'ACM':
		data,data_PAP, data_PSP, train_mask, val_mask, test_mask = load_ACM(seed=args.seed, path='')
		in_dim = data_PAP.x.shape[1]
		num_class = data_PAP.y.unique().shape[0] 
		# print(f'y {data_PAP.y.shape}')
		# print(f'class {num_class}')
		data = data.to(args.device)
		data_PAP = data_PAP.to(args.device)
		data_PSP = data_PSP.to(args.device)
		graphs = [data_PAP,data_PSP]

	elif args.dataset == 'Urban':
		data, data_ZOZ, data_ZTZ,data_ZThZ, train_mask, val_mask, test_mask = load_Urban(seed=args.seed,path="")
		in_dim = data_ZOZ.x.shape[1]
		num_class = data_ZOZ.y.unique().shape[0]
		#print('y',num_class)
		data_ZOZ = data_ZOZ.to(args.device)
		data_ZTZ = data_ZTZ.to(args.device)
		data_ZThZ = data_ZThZ.to(args.device)
		# data_PAIAP = data_PAIAP.to(args.device)
		graphs = [data_ZOZ,data_ZTZ,data_ZThZ]


	num_model = args.num_model
	if args.dataset == 'Company':
		model = MultiGCN(in_dim=in_dim, hidden=args.hidden, out_dim=num_class,attention_dim=args.attention_dim, num_gcn=num_model, dropout=args.dropout, layer_s=args.layer_size, num_path=len(graphs)).double().to(args.device)
	else:
		#model = FilterGCN(in_dim=in_dim, hidden=args.hidden, out_dim=num_class, num_gcn=num_model, dropout=args.dropout, layer_s=args.layer_size, num_path=len(graphs)).to(args.device)
		model = MultiGCN(in_dim=in_dim, hidden=args.hidden, out_dim=num_class,attention_dim=args.attention_dim, num_gcn=num_model, dropout=args.dropout, layer_s=args.layer_size, num_path=len(graphs)).to(args.device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	train_mask = train_mask.to(args.device)
	val_mask = val_mask.to(args.device)
	test_mask = test_mask.to(args.device)
	#loss_fcn = torch.nn.CrossEntropyLoss()
	loss_fcn = torch.nn.NLLLoss()
	best_val = 0
	patience = 0
	# torch.save(model.state_dict(), f'model_state_dict_minmax_{args.dataset}_{args.seed}_{count}.pth')
	model_state = copy.deepcopy(model.state_dict())
	print("training...")
	if args.inference:
		for epoch in range(args.epochs):

			model.train()
			optimizer.zero_grad()
			predict, std, out = model(graphs)
			# print(f'predict:{predict.shape}')
			# print(f'predict:{predict[train_mask].shape}')
			# print(f'y:{graphs[0].y[train_mask].shape}')
			loss = loss_fcn(predict[train_mask], graphs[0].y[train_mask])
			#cov_norm = torch.norm(std, p='fro') **2  # Frobenius norm squared
			cov_norm = torch.norm(std, p=1) ** 2

			lambda_cov = args.lambda_cov  # Regularization strength
			loss += lambda_cov * cov_norm

			loss.backward()
			optimizer.step()
			
			val = (torch.max(predict[val_mask],dim=1)[1] == graphs[0].y[val_mask]).float().mean()
			if val>best_val:
				#print(f'refresh')
				model_state = copy.deepcopy(model.state_dict())
				best_val = val
				std_mean = std
				patience =0
			else:
				patience+=1
			if patience>=args.patience:
				break
			#print("patience:",patience)
			#print("best_val:",best_val)
			#print(torch.stack(std_mean).shape)
		
			#print(std_mean)
			#torch.save(model_state,'best_model_state_dict_minmax_MLP_'+str(args.dataset)+'_'+str(args.seed)+'.pth')
			torch.save(model_state,'model_state_dict_minmax_'+str(args.dataset)+'_'+str(args.seed)+'_'+str(count)+'.pth')
	#model.load_state_dict(torch.load('model_state_dict_minmax_'+str(args.dataset)+'_'+str(args.seed)+'_'+str(count)+'.pth'))
	print("loading...")
	#model.load_state_dict(torch.load('best_model_state_dict_minmax_MLP_'+str(args.dataset)+'_'+str(args.seed)+'.pth'))
	#model.load_state_dict(torch.load('best_model_state_dict_minmax_MLPSAGE_'+str(args.dataset)+'_'+str(args.seed)+'.pth'))
	model.load_state_dict(torch.load('model_state_dict_minmax_'+str(args.dataset)+'_'+str(args.seed)+'_'+str(count)+'.pth'))
	#cov_mean = (torch.stack(std_mean)).mean(dim=0)
	#print(cov_mean.shape)
	#torch.save(cov_mean,'cov_mean7_val.pt')
	@torch.no_grad()
	def test(model):
		model.eval()

		predict, std, out = model(graphs)
		#predict_auc = model.forward_predict(graphs)
		loss = loss_fcn(predict[test_mask],graphs[0].y[test_mask])

		#print("test_loss:",loss.item())
		acc = accuracy_score(graphs[0].y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		test_report= classification_report(graphs[0].y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		#print("acc:",acc)
		#print("test report",test_report)

		y_true = graphs[0].y[test_mask].cpu().numpy()
		n_classes = predict.size(1)
		y_true_bin = label_binarize(y_true, classes=range(n_classes))
		y_pred_labels = predict[test_mask].cpu().numpy()
		#y_pred_bin = label_binarize(y_pred_labels, classes=range(n_classes))
		if args.dataset=='Company':
			y_true_bin = np.zeros((y_true.shape[0], 2))
			y_true_bin[np.arange(y_true.shape[0]), y_true] = 1
		auc = roc_auc_score(y_true_bin, y_pred_labels, multi_class='ovr')
		label = torch.max(predict[test_mask],dim=1)[1].cpu().numpy()
		ground_truth = graphs[0].y[test_mask].cpu().numpy()
		correct_multi = (label == ground_truth)
		out = out[test_mask]
		#auc = roc_auc_score(graphs[0].y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy(),multi_class='ovo')
		#print("auc:",auc)
		#print('std',std)
		#cov_mean = (torch.stack(std_mean)).mean(dim=0)
		#torch.save(cov_mean,'cov_mean7_test.pt')
		# out = out[test_mask].cpu().numpy()

		return correct_multi,out,auc,acc
		#return auc,acc

		#print('Done')
	correct_multi,out,auc,acc = test(model)
	#auc,acc = test(model)
	#delete_file(f'model_state_dict_minmax_{args.dataset}_{args.seed}.pth')
	print("normal condition")
	print('acc',acc)
	print('auc',auc)
	print("-----------------------")
	# #case 1, acc average
	# acc_GCN = []
	# auc_GCN = []
	# predictions = model.forward_single_gcn(graphs)
	# for i in range(len(graphs)*num_model):
	# 	predict = predictions[i]
	# 	y_true = graphs[0].y[test_mask].cpu().numpy()
	# 	n_classes = predict.size(1)
	# 	y_true_bin = label_binarize(y_true, classes=range(n_classes))
	# 	if args.dataset=='Company':
	# 		y_true_bin = np.zeros((y_true.shape[0], 2))
	# 		y_true_bin[np.arange(y_true.shape[0]), y_true] = 1
	# 	pred_labels = F.softmax(predict[test_mask], dim=1).cpu().numpy()
	# 	single_auc = roc_auc_score(y_true_bin, pred_labels, multi_class='ovr')
	# 	single_acc = accuracy_score(graphs[0].y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
	# 	acc_GCN.append(single_acc)
	# 	auc_GCN.append(single_auc)
	# print(acc_GCN)
	# # case 2, y average, then compute one acc,auc
	# predictions_total = torch.stack(predictions)
	# final_predictions = torch.sum(predictions_total, dim=0)
	# #print(final_predictions.shape)
	# acc = accuracy_score(graphs[0].y[test_mask].cpu().numpy(),torch.max(final_predictions[test_mask],dim=1)[1].cpu().numpy())
	# test_report= classification_report(graphs[0].y[test_mask].cpu().numpy(),torch.max(final_predictions[test_mask],dim=1)[1].cpu().numpy())
	# #print(test_report)
	# y_true = graphs[0].y[test_mask].cpu().numpy()
	# n_classes = final_predictions.size(1)
	# y_true_bin = label_binarize(y_true, classes=range(n_classes))
	# if args.dataset=='Company':
	# 		y_true_bin = np.zeros((y_true.shape[0], 2))
	# 		y_true_bin[np.arange(y_true.shape[0]), y_true] = 1
	# y_pred_labels = F.softmax(final_predictions[test_mask], dim=1).cpu().numpy()
	# #y_pred_bin = label_binarize(y_pred_labels, classes=range(n_classes))
	# auc = roc_auc_score(y_true_bin, y_pred_labels, multi_class='ovr')
	# print("average y, one acc----------")
	# print('acc',acc)
	# print('auc',auc)
	# print("average acc,auc----------")
	# print('GNN average acc',sum(acc_GCN)/len(acc_GCN))
	# print('GNN average auc',sum(auc_GCN)/len(auc_GCN))

	return correct_multi,out
	#return (auc,acc,model.state_dict())

def run_MGCN(args):
	if args.dataset == 'Company':
		data,data_COC, data_CPC, train_mask, val_mask, test_mask = load_company(seed=args.seed, dataname='',path='')
		in_dim = data_COC.x.shape[1]
		num_class = data_COC.y.unique().shape[0] 
		data = data.to(args.device)
		data_COC = data_COC.to(args.device)
		data_CPC = data_CPC.to(args.device)
		graphs = [data_COC,data_CPC]
	elif args.dataset == 'DBLP':

		data, data_APA, data_APCPA, data_APTPA, train_mask, val_mask, test_mask = load_DBLP(seed=args.seed, dataname='DBLP',path='')
		in_dim = data_APA.x.shape[1]
		num_class = data_APA.y.unique().shape[0]
		data = data.to(args.device)
		data_APA = data_APA.to(args.device)
		data_APCPA = data_APCPA.to(args.device)
		data_APTPA = data_APTPA.to(args.device)
		graphs = [data_APA,data_APTPA,data_APCPA]
	elif args.dataset == 'IMDB':

		data,data_MAM, data_MDM,data_AMA, data_DMD, train_mask, val_mask, test_mask = load_IMDB(seed=args.seed, path='')
		in_dim = data_MAM.x.shape[1]
		num_class = data_MAM.y.unique().shape[0] 
		data = data.to(args.device)
		data_MAM = data_MAM.to(args.device)
		data_MDM = data_MDM.to(args.device)
		data_AMA = data_AMA.to(args.device)
		data_DMD = data_DMD.to(args.device)
		graphs = [data_MAM,data_MDM,data_AMA,data_DMD]
	elif args.dataset == 'ACM':
		data,data_PAP, data_PSP, train_mask, val_mask, test_mask = load_ACM(seed=args.seed, path='')
		in_dim = data_PAP.x.shape[1]
		num_class = data_PAP.y.unique().shape[0] 
		# print(f'y {data_PAP.y.shape}')
		# print(f'class {num_class}')
		data = data.to(args.device)
		data_PAP = data_PAP.to(args.device)
		data_PSP = data_PSP.to(args.device)
		graphs = [data_PAP,data_PSP]

	elif args.dataset == 'Urban':
		data, data_ZOZ, data_ZTZ,data_ZThZ, train_mask, val_mask, test_mask = load_Urban(seed=args.seed,path="")
		in_dim = data_ZOZ.x.shape[1]
		num_class = data_ZOZ.y.unique().shape[0]
		#print('y',num_class)
		data_ZOZ = data_ZOZ.to(args.device)
		data_ZTZ = data_ZTZ.to(args.device)
		data_ZThZ = data_ZThZ.to(args.device)
		# data_PAIAP = data_PAIAP.to(args.device)
		graphs = [data_ZOZ,data_ZTZ,data_ZThZ]

	num_model = args.num_model
	GCN_lists = [[] for i in range(num_model)]
	#print("GCN_List is:",GCN_lists)
	#SAGE_list = []

	num_metapath = len(graphs)
	for i in range(num_metapath):
		for j in range(num_model):
			if args.dataset == 'Company':
				GCN_lists[j].append(GCN(in_dim=in_dim, hidden=args.hidden, out_dim=num_class, dropout=args.dropout, layer_s=args.layer_size).double().to(args.device))
				#GCN_lists[j].append(GraphSAGE(in_dim=in_dim, hidden=args.hidden, out_dim=num_class, dropout=args.dropout,layer_s=args.layer_size).double().to(args.device))
				#GCN_lists[j].append(GAT(in_dim=in_dim, hidden=args.hidden, out_dim=num_class, dropout=args.dropout,layer_s=args.layer_size).double().to(args.device))
			else:
				GCN_lists[j].append(GCN(in_dim=in_dim, hidden=args.hidden, out_dim=num_class, dropout=args.dropout, layer_s=args.layer_size).to(args.device))
				#GCN_lists[j].append(GraphSAGE(in_dim=in_dim, hidden=args.hidden, out_dim=num_class, dropout=args.dropout,layer_s=args.layer_size).to(args.device))
				#GCN_lists[j].append(GAT(in_dim=in_dim, hidden=args.hidden, out_dim=num_class, dropout=args.dropout,layer_s=args.layer_size).to(args.device))
	# 10 * 3 GCN_lists shape
	#print(len(GCN_lists[0]))
	#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	train_mask = train_mask.to(args.device)
	val_mask = val_mask.to(args.device)
	test_mask = test_mask.to(args.device)
	loss_fcn = torch.nn.CrossEntropyLoss()
	best_val_list = [[0 for j in range(num_metapath)] for i in range(num_model)]
	patience_list = [[0 for j in range(num_metapath)] for i in range(num_model)]
	predictions = []
	# ex: 5 gcn, 3 meta-path, 15 total gcns
	if args.inference:
		for i in range(len(GCN_lists)):
			for j in range(num_metapath):
				for epoch in range(args.epochs):
					GCN_lists[i][j].train()
					optimizer = torch.optim.Adam(GCN_lists[i][j].parameters(), lr=args.lr, weight_decay=args.weight_decay)
					optimizer.zero_grad()
					predict = GCN_lists[i][j](graphs[j].x, graphs[j].edge_index)

					loss = loss_fcn(predict[train_mask], graphs[0].y[train_mask])
					loss.backward()
					optimizer.step()

					val = (torch.max(predict[val_mask],dim=1)[1] == graphs[0].y[val_mask]).float().mean()

					if val>best_val_list[i][j]:
						# file_path = 'model_state_dict_' + str(i) + '_' + str(j) + '.pth'
						# if os.path.exists(file_path):
						# 	os.remove(file_path)  # Remove the existing file
						#torch.save(GCN_lists[i][j].state_dict(), 'model_state_dict_'+str(i)+'_'+str(j)+'.pth')
						torch.save(GCN_lists[i][j].state_dict(), 'model_state_dict_SAGE'+str(args.seed)+'_'+str(i)+'_'+str(j)+'.pth')
						best_val_list[i][j]=val
						patience_list[i][j]=0
					else:
						patience_list[i][j]+=1
					if patience_list[i][j]>=args.patience:
						break
					#print("patience:",patience_list[i][j])
					#print("best_val:",best_val_list[i][j])
	@torch.no_grad()
	def test(model,j):
		model.eval()

		m_index = j
		predict = model(graphs[m_index].x, graphs[m_index].edge_index)
		loss = loss_fcn(predict[test_mask],graphs[0].y[test_mask])
		#print("test_loss:",loss.item())

		#acc = accuracy_score(data_APA.y[test_mask].cpu().numpy(),torch.max(final_predictions[test_mask],dim=1)[1].cpu().numpy())
		#test_report= classification_report(data_APA.y[test_mask].cpu().numpy(),torch.max(final_predictions[test_mask],dim=1)[1].cpu().numpy())
		# print("acc:",acc)
		# print("test report",test_report)
		return predict
	acc_GCN = []
	auc_GCN = []
	for i in range(len(GCN_lists)):
		for j in range(num_metapath):
			#GCN_lists[i][j].load_state_dict(torch.load('model_state_dict_'+str(i)+'_'+str(j)+'.pth'))
			GCN_lists[i][j].load_state_dict(torch.load('model_state_dict_SAGE'+str(args.seed)+'_'+str(i)+'_'+str(j)+'.pth'))
			predict = test(GCN_lists[i][j],j)
			y_true = graphs[0].y[test_mask].cpu().numpy()
			n_classes = predict.size(1)
			y_true_bin = label_binarize(y_true, classes=range(n_classes))
			pred_labels = F.softmax(predict[test_mask], dim=1).cpu().numpy()
			if args.dataset=='Company':
				y_true_bin = np.zeros((y_true.shape[0], 2))
				y_true_bin[np.arange(y_true.shape[0]), y_true] = 1
			single_auc = roc_auc_score(y_true_bin, pred_labels, multi_class='ovr')
			single_acc = accuracy_score(graphs[0].y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
			predictions.append(predict)

			acc_GCN.append(single_acc)
			auc_GCN.append(single_auc)
	predictions_total = torch.stack(predictions)
	final_predictions = torch.sum(predictions_total, dim=0)
	#print(final_predictions.shape)
	acc = accuracy_score(graphs[0].y[test_mask].cpu().numpy(),torch.max(final_predictions[test_mask],dim=1)[1].cpu().numpy())
	test_report= classification_report(graphs[0].y[test_mask].cpu().numpy(),torch.max(final_predictions[test_mask],dim=1)[1].cpu().numpy())
	print(test_report)
	y_true = graphs[0].y[test_mask].cpu().numpy()
	n_classes = final_predictions.size(1)
	y_true_bin = label_binarize(y_true, classes=range(n_classes))
	y_pred_labels = F.softmax(final_predictions[test_mask], dim=1).cpu().numpy()
	if args.dataset=='Company':
		y_true_bin = np.zeros((y_true.shape[0], 2))
		y_true_bin[np.arange(y_true.shape[0]), y_true] = 1
	#y_pred_bin = label_binarize(y_pred_labels, classes=range(n_classes))
	auc = roc_auc_score(y_true_bin, y_pred_labels, multi_class='ovr')
	print('acc',acc)
	print('auc',auc)
	print('GCN average acc',sum(acc_GCN)/len(acc_GCN))
	print('GCN average auc',sum(auc_GCN)/len(auc_GCN))

	return auc,acc

def run_GCN(args):
	data, data_APA, data_APCPA, train_mask, val_mask, test_mask = load_IMDB(seed=0,path='')
	in_dim = data_APA.x.shape[1]
	num_class = data_APA.y.unique().shape[0] 
	model = GCN(in_dim=in_dim, hidden=args.hidden, out_dim=num_class, dropout=args.dropout, layer_s=args.layer_size).to(args.device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	data_APA = data_APA.to(args.device)
	data_APCPA = data_APCPA.to(args.device)
	loss_fcn = torch.nn.CrossEntropyLoss()
	best_val = 0
	patience = 0
	graphs = [data_APA.edge_index,data_APCPA.edge_index]
	selected_graph = graphs[args.choice]
	for epoch in range(args.epochs):
		model.train()
		optimizer.zero_grad()
		predict = model(data_APA.x, selected_graph)
		loss = loss_fcn(predict[train_mask], data_APA.y[train_mask])
		loss.backward()
		optimizer.step()

		val = (torch.max(predict[val_mask],dim=1)[1] == data_APA.y[val_mask]).float().mean()
		if val>best_val:
			torch.save(model.state_dict(), 'model_state_dict.pth')
			best_val = val
			patience =0
		else:
			patience+=1
		if patience>=args.patience:
			break
		print("patience:",patience)
		print("best_val:",best_val)
	model.load_state_dict(torch.load('model_state_dict.pth'))
	@torch.no_grad()
	def test(model):
		model.eval()

		predict = model(x=data_APA.x, g=selected_graph)

		loss = loss_fcn(predict[test_mask],data_APA.y[test_mask])
		print("test_loss:",loss.item())
		acc = accuracy_score(data_APA.y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		test_report= classification_report(data_APA.y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		print("acc:",acc)
		print("test report",test_report)
		return acc

		print('Done')
	auc = test(model)

	return model

def run_HAN(args):
	if args.dataset == 'Company':
		data,data_COC, data_CPC, train_mask, val_mask, test_mask = load_company(seed=args.seed, dataname='',path='')
		in_dim = data_COC.x.shape[1]
		num_class = data['company'].y.unique().shape[0] 
		data = data.to(args.device)
		data_COC = data_COC.to(args.device)
		data_CPC = data_CPC.to(args.device)
		author = data['company']
		graphs = [data_COC,data_CPC]
		train_mask = train_mask.to(args.device)
		val_mask = val_mask.to(args.device)
		test_mask = test_mask.to(args.device)
		meta_data = data.metadata()


	elif args.dataset == 'DBLP':
		data,data_APA, data_APCPA, data_APTPA, train_mask, val_mask, test_mask = load_DBLP(seed=args.seed, dataname='DBLP',path='')
		print("Success")
		in_dim = data_APA.x.shape[1]
		num_class = data['author'].y.unique().shape[0]
		data = data.to(args.device)
		data_APA = data_APA.to(args.device)
		data_APCPA = data_APCPA.to(args.device)
		data_APTPA = data_APTPA.to(args.device)
		author = data['author']
		graphs = [data_APA,data_APCPA,data_APTPA]
		train_mask = train_mask.to(args.device)
		val_mask = val_mask.to(args.device)
		test_mask = test_mask.to(args.device)
		meta_data = data.metadata()
	elif args.dataset == 'IMDB':
		data,data_MAM, data_MDM,data_AMA, data_DMD, train_mask, val_mask, test_mask = load_IMDB(seed=args.seed, path='')
		in_dim = data_MAM.x.shape[1]
		num_class = data_MAM.y.unique().shape[0] 
		data = data.to(args.device)
		data_MAM = data_MAM.to(args.device)
		data_MDM = data_MDM.to(args.device)
		data_AMA = data_AMA.to(args.device)
		data_DMD = data_DMD.to(args.device)
		graphs = [data_MAM,data_MDM,data_AMA,data_DMD]
		author = data['movie']
		train_mask = train_mask.to(args.device)
		val_mask = val_mask.to(args.device)
		test_mask = test_mask.to(args.device)
		meta_data = data.metadata()
		print('meta',meta_data)
	elif args.dataset == 'ACM':
		data,data_PAP, data_PSP, train_mask, val_mask, test_mask = load_ACM(seed=args.seed, path='')
		in_dim = data_PAP.x.shape[1]
		num_class = data_PAP.y.unique().shape[0] 
		data = data.to(args.device)
		data_PAP = data_PAP.to(args.device)
		data_PSP = data_PSP.to(args.device)
		author = data['paper']
		graphs = [data_PAP,data_PSP]
		train_mask = train_mask.to(args.device)
		val_mask = val_mask.to(args.device)
		test_mask = test_mask.to(args.device)
		meta_data = data.metadata()
		print('meta',meta_data)

	elif args.dataset == 'Urban':
		data, data_ZOZ, data_ZTZ,data_ZThZ, train_mask, val_mask, test_mask = load_Urban(seed=args.seed,path="")
		in_dim = data_ZOZ.x.shape[1]
		num_class = data_ZOZ.y.unique().shape[0]
		#print('y',num_class)
		data = data.to(args.device)
		data_ZOZ = data_ZOZ.to(args.device)
		data_ZTZ = data_ZTZ.to(args.device)
		data_ZThZ = data_ZThZ.to(args.device)
		author = data['0']
		train_mask = train_mask.to(args.device)
		val_mask = val_mask.to(args.device)
		test_mask = test_mask.to(args.device)
		meta_data = data.metadata()
		# data_PAIAP = data_PAIAP.to(args.device)
		graphs = [data_ZOZ,data_ZTZ,data_ZThZ]

	# print('edge',data.edge_index_dict)
	# print('x_dict',data.x_dict)
	if args.dataset == 'Company':
		#model = HAN(in_channels=-1, hidden_channels=args.hidden,out_channels=num_class,author_data=meta_data).double().to(args.device)
		model = basic_HAN(in_dim=in_dim, hidden=args.hidden,atten_dim=args.att,out_dim=num_class,metadata=meta_data).double().to(args.device)
	else:
		#model = HAN(in_channels=-1, hidden_channels=args.hidden,out_channels=num_class,author_data=meta_data).to(args.device)
		model = basic_HAN(in_dim=in_dim, hidden=args.hidden,atten_dim=args.att,out_dim=num_class,metadata=meta_data).to(args.device)

	with torch.no_grad():  # Initialize lazy modules.
		out = model(data.x_dict, data.edge_index_dict)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

	loss_fcn = torch.nn.NLLLoss()
	#loss_fcn = torch.nn.CrossEntropyLoss()
	best_val = 0
	patience = 0
	heads=8
	# print("data feature dict",data.edge_index_dict)

	for epoch in range(args.epochs):
		model.train()
		optimizer.zero_grad()
		predict = model(data.x_dict, data.edge_index_dict)[0]
		#predict, out = model(data.x_dict, data.edge_index_dict)
		#print("predict HAN",predict.shape)
		#print(author.y.shape)
		loss = loss_fcn(predict[train_mask], author.y[train_mask])
		loss.backward()
		optimizer.step()
		y_true = author.y[val_mask].cpu().numpy()
		n_classes = predict.size(1)
		# y_true_bin = label_binarize(y_true, classes=range(n_classes))
		# y_pred_labels = torch.max(predict[val_mask], dim=1)[1].cpu().numpy()
		# y_pred_bin = label_binarize(y_pred_labels, classes=range(n_classes))
		# val = roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovr')
		val = (torch.max(predict[val_mask],dim=1)[1] == author.y[val_mask]).float().mean()
		if val>best_val:
			torch.save(model.state_dict(), 'HAN_model_state_dict'+str(args.dataset)+str(args.seed)+'.pth')
			best_val = val
			patience =0
		else:
			patience+=1
		if patience>=args.patience:
			break
		#print("patience:",patience)
		#print("best_val:",best_val)
	model.load_state_dict(torch.load('HAN_model_state_dict'+str(args.dataset)+str(args.seed)+'.pth'))
	@torch.no_grad()
	def test(model):
		model.eval()

		predict = model(data.x_dict, data.edge_index_dict)[0]
		#predict,out = model(data.x_dict, data.edge_index_dict)
		loss = loss_fcn(predict[test_mask],author.y[test_mask])
		#print("test_loss:",loss.item())
		acc = accuracy_score(author.y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		test_report= classification_report(author.y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		print("acc:",acc)
		#print("test report",test_report)

		y_true = graphs[0].y[test_mask].cpu().numpy()
		n_classes = predict.size(1)
		y_true_bin = label_binarize(y_true, classes=range(n_classes))
		y_pred_labels = F.softmax(predict[test_mask], dim=1).cpu().numpy()
		#y_pred_bin = label_binarize(y_pred_labels, classes=range(n_classes))
		if args.dataset=='Company':
			y_true_bin = np.zeros((y_true.shape[0], 2))
			y_true_bin[np.arange(y_true.shape[0]), y_true] = 1
		auc = roc_auc_score(y_true_bin, y_pred_labels, multi_class='ovr')
		#auc = roc_auc_score(data_COC.y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		print("auc:",auc)
		# print(out['author'].shape)
		# out = out['author'][test_mask].cpu().numpy()
		label = torch.max(predict[test_mask],dim=1)[1].cpu().numpy()
		ground_truth = author.y[test_mask].cpu().numpy()
		correct_han = (label == ground_truth)

		# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
		# reduced_embeddings = tsne.fit_transform(out)  # Shape: [num_nodes, 2]

		# # Create scatter plot
		# plt.figure(figsize=(10, 8))
		# scatter = plt.scatter(
		#     reduced_embeddings[:, 0], reduced_embeddings[:, 1],
		#     c=y_true, cmap='Accent', alpha=0.7
		# )
		# #plt.colorbar(scatter, label='Class Labels')
		# plt.title('Visualization of Node Embeddings DBLP (3 Classes)')
		# plt.grid(True)
		# plt.show()
		return acc,correct_han

		#print('Done')
	acc,correct_han = test(model)

	return correct_han

def run_SeHGNN(args,count):

	if args.dataset == 'Company':
		data,data_COC, data_CPC, train_mask, val_mask, test_mask = load_company(seed=args.seed, dataname='',path='')
		in_dim = data_COC.x.shape[1]
		num_class = data_COC.y.unique().shape[0] 
		data = data.to(args.device)
		data_COC = data_COC.to(args.device)
		data_CPC = data_CPC.to(args.device)
		graphs = [data_COC,data_CPC]
	elif args.dataset == 'DBLP':

		data, data_APA, data_APCPA, data_APTPA, train_mask, val_mask, test_mask = load_DBLP(seed=args.seed, dataname='DBLP',path='')
		in_dim = data_APA.x.shape[1]
		num_class = data_APA.y.unique().shape[0]
		data_APA = data_APA.to(args.device)
		data_APCPA = data_APCPA.to(args.device)
		data_APTPA = data_APTPA.to(args.device)
		graphs = [data_APA,data_APTPA,data_APCPA]
	elif args.dataset == 'IMDB':

		data,data_MAM, data_MDM,data_AMA, data_DMD, train_mask, val_mask, test_mask = load_IMDB(seed=args.seed, path='')
		in_dim = data_MAM.x.shape[1]
		num_class = data_MAM.y.unique().shape[0] 
		data = data.to(args.device)
		data_MAM = data_MAM.to(args.device)
		data_MDM = data_MDM.to(args.device)
		data_AMA = data_AMA.to(args.device)
		data_DMD = data_DMD.to(args.device)
		graphs = [data_MAM,data_MDM,data_AMA,data_DMD]
	elif args.dataset == 'ACM':
		data,data_PAP, data_PSP, train_mask, val_mask, test_mask = load_ACM(seed=args.seed, path='')
		in_dim = data_PAP.x.shape[1]
		num_class = data_PAP.y.unique().shape[0] 
		# print(f'y {data_PAP.y.shape}')
		# print(f'class {num_class}')
		data = data.to(args.device)
		data_PAP = data_PAP.to(args.device)
		data_PSP = data_PSP.to(args.device)
		graphs = [data_PAP,data_PSP]


	elif args.dataset == 'Urban':
		data, data_ZOZ, data_ZTZ,data_ZThZ, train_mask, val_mask, test_mask = load_Urban(seed=args.seed,path="")
		in_dim = data_ZOZ.x.shape[1]
		num_class = data_ZOZ.y.unique().shape[0]
		#print('y',num_class)
		data_ZOZ = data_ZOZ.to(args.device)
		data_ZTZ = data_ZTZ.to(args.device)
		data_ZThZ = data_ZThZ.to(args.device)
		# data_PAIAP = data_PAIAP.to(args.device)
		graphs = [data_ZOZ,data_ZTZ,data_ZThZ]

	adjs = [graphs[i].edge_index for i in range(len(graphs))]
	num_model = args.num_model
	if args.dataset == 'Company':
		model = SeHGNN(in_dim=in_dim, hid_dim=args.hidden, out_dim=num_class, num_heads=2, dropout=args.dropout,layer_s=args.layer_size).double().to(args.device)
	elif args.dataset == 'Urban' or args.dataset == 'DBLP':
		model = SeHGNN(in_dim=in_dim, hid_dim=args.hidden, out_dim=num_class, num_heads=3, dropout=args.dropout,layer_s=args.layer_size).to(args.device)
	elif args.dataset == 'IMDB':
		model = SeHGNN(in_dim=in_dim, hid_dim=args.hidden, out_dim=num_class, num_heads=2, dropout=args.dropout,layer_s=args.layer_size).to(args.device)
	else:
		model = SeHGNN(in_dim=in_dim, hid_dim=args.hidden, out_dim=num_class, num_heads=2, dropout=args.dropout,layer_s=args.layer_size).to(args.device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	train_mask = train_mask.to(args.device)
	val_mask = val_mask.to(args.device)
	test_mask = test_mask.to(args.device)
	loss_fcn = torch.nn.CrossEntropyLoss()
	best_val = 0
	patience = 0
	# torch.save(model.state_dict(), f'model_state_dict_minmax_{args.dataset}_{args.seed}_{count}.pth')
	model_state = copy.deepcopy(model.state_dict())
	for epoch in range(args.epochs):

		model.train()
		optimizer.zero_grad()
		predict = model(adjs,graphs[0].x)
		# print(f'predict:{predict.shape}')
		# print(f'predict:{predict[train_mask].shape}')
		# print(f'y:{graphs[0].y[train_mask].shape}')
		loss = loss_fcn(predict[train_mask], graphs[0].y[train_mask])

		loss.backward()
		optimizer.step()
		
		val = (torch.max(predict[val_mask],dim=1)[1] == graphs[0].y[val_mask]).float().mean()
		if val>best_val:
			#print(f'refresh')
			model_state = copy.deepcopy(model.state_dict())
			best_val = val
			patience =0
			#print("val:",val)
		else:
			patience+=1
		if patience>=args.patience:
			break

	torch.save(model_state,'model_state_dict_SeHGNN_'+str(args.dataset)+'_'+str(args.seed)+'_'+str(count)+'.pth')
	model.load_state_dict(torch.load('model_state_dict_SeHGNN_'+str(args.dataset)+'_'+str(args.seed)+'_'+str(count)+'.pth'))
	#cov_mean = (torch.stack(std_mean)).mean(dim=0)
	#print(cov_mean.shape)
	#torch.save(cov_mean,'cov_mean7_val.pt')
	@torch.no_grad()
	def test(model):
		model.eval()

		predict = model(adjs,graphs[0].x)

		loss = loss_fcn(predict[test_mask],graphs[0].y[test_mask])

		#print("test_loss:",loss.item())
		acc = accuracy_score(graphs[0].y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		test_report= classification_report(graphs[0].y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy())
		#print("acc:",acc)
		#print("test report",test_report)

		y_true = graphs[0].y[test_mask].cpu().numpy()
		n_classes = predict.size(1)
		y_true_bin = label_binarize(y_true, classes=range(n_classes))
		y_pred_labels = F.softmax(predict[test_mask], dim=1).cpu().numpy()
		#y_pred_bin = label_binarize(y_pred_labels, classes=range(n_classes))
		if args.dataset=='Company':
			y_true_bin = np.zeros((y_true.shape[0], 2))
			y_true_bin[np.arange(y_true.shape[0]), y_true] = 1
		auc = roc_auc_score(y_true_bin, y_pred_labels, multi_class='ovr')
		#auc = roc_auc_score(graphs[0].y[test_mask].cpu().numpy(),torch.max(predict[test_mask],dim=1)[1].cpu().numpy(),multi_class='ovo')
		print("acc:",acc)
		print('auc',auc)
		#cov_mean = (torch.stack(std_mean)).mean(dim=0)
		#torch.save(cov_mean,'cov_mean7_test.pt')
		return auc,acc

		#print('Done')
	auc,acc = test(model)
	#delete_file(f'model_state_dict_minmax_{args.dataset}_{args.seed}.pth')
	return (auc,acc,model.state_dict())


def main(args):
	#run_SeHGNN(args,count=-1)
	#run_MGCN(args)
	#run_Multi(args,count=-1)
	#run_SeHGNN(args,count=-1)
	#run_GCN(args)

if __name__ == "__main__":
	parser = argparse.ArgumentParser("HAN")
	parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='gpu or cpu')
	parser.add_argument('--epochs', type=int, default=500)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=0.0005)
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--dropout1', type=float, default=0)
	parser.add_argument('--dropout2', type=float, default=0)
	parser.add_argument('--patience',type=int,default=50)
	parser.add_argument('--num_model',type=int,default=3)
	parser.add_argument('--seed',type=int,default=0)
	parser.add_argument('--hidden',type=int,default=64)
	parser.add_argument('--hidden2',type=int,default=32)
	parser.add_argument('--hidden3',type=int,default=64)
	parser.add_argument('--layer_size',type=int,default=2)
	parser.add_argument('--layer_size2',type=int,default=2)
	parser.add_argument('--layer_size3',type=int,default=2)
	parser.add_argument('--choice',type=int,default=0)
	parser.add_argument('--att',type=int,default=10)
	parser.add_argument('--lambda_cov', type=float, default=0)
	parser.add_argument('--dataset',type=str,default='DBLP')
	parser.add_argument('--fused',type=bool,default=False)
	parser.add_argument('--attention_dim',type=int,default=8)
	parser.add_argument('--inference', default=True, action='store_false', help='Bool type')

	args = parser.parse_args()
	main(args)
