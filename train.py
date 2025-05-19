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
	return correct_multi,out
	#return (auc,acc,model.state_dict())

def main(args):
	#run_Multi(args,count=-1)


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
