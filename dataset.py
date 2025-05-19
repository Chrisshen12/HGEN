# import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_networkx, subgraph, degree
from torch_geometric.datasets import Planetoid, DBLP, IMDB, HGBDataset, LastFM, MovieLens1M, AMiner, OGB_MAG
import networkx as nx
import numpy as np
import random
import torch_geometric.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv
import torch_geometric as geo

def get_binary_mask(total_size, indices):
	mask = torch.zeros(total_size)
	mask[indices] = 1
	return mask.byte().to(torch.bool)

def load_DBLP(seed=42,dataname='a',path=''):
	if path != '':
		os.chdir(path)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
	    torch.cuda.manual_seed(seed)
	metapaths = [[('author', 'paper'), ('paper', 'author')],
             [('author', 'paper'),('paper','term'),('term','paper'), ('paper', 'author')],
             [('author', 'paper'),('paper','conference'),('conference','paper'), ('paper', 'author')]]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                           drop_unconnected_node_types=True)
	if (dataname=="DBLP"):
		dataset = DBLP(root='tmp/DBLP',transform=transform)[0]
		#print(dataset['author', 'metapath_0', 'author'].edge_index)
		#dataset2 = DBLP(root='tmp/DBLP2')[0]
		#print(dataset2)
		x = dataset['author'].x
		#print("x",x.shape)
		y = dataset['author'].y
		# APA = torch.load('combined_author.pt')['APA']
		# APTPA = torch.load('combined_author.pt')['APTPA']
		# APCPA = torch.load('combined_author.pt')['APCPA']
		APA = dataset['author', 'metapath_0', 'author'].edge_index
		APTPA = dataset['author', 'metapath_1', 'author'].edge_index
		APCPA = dataset['author', 'metapath_2', 'author'].edge_index
		# print("APA",APA.shape)
		# print("APTPA",APTPA.shape)
		# print("APCPA",APCPA.shape)
		data_APA = Data(x=x, edge_index=APA, y=y)
		data_APTPA = Data(x=x, edge_index=APTPA, y=y)
		data_APCPA = Data(x=x, edge_index=APCPA, y=y)
		# data_APA = Data(x=x, edge_index=dense_to_sparse(APA.fill_diagonal_(1))[0], y=y)
		# data_APTPA = Data(x=x, edge_index=dense_to_sparse(APTPA.fill_diagonal_(1))[0], y=y)
		# data_APCPA = Data(x=x, edge_index=dense_to_sparse(APCPA.fill_diagonal_(1))[0], y=y)
		# print(f'APA:{len(data_APA.edge_index[0])}')
		# print(f'APTPA:{len(data_APTPA.edge_index[0])}')
		# print(f'APCPA:{len(data_APCPA.edge_index[0])}')

		# num_nodes = x.shape[0]
		# float_mask = np.random.permutation(np.linspace(0, 1, num_nodes))

		# train_idx = np.where(float_mask <= 0.2)[0]
		# val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
		# test_idx = np.where(float_mask > 0.3)[0]
		# train_mask = get_binary_mask(num_nodes, train_idx)
		# val_mask = get_binary_mask(num_nodes, val_idx)
		# test_mask = get_binary_mask(num_nodes, test_idx)

		# np.save(f'train_mask_{seed}', train_mask.numpy())
		# np.save(f'test_mask_{seed}', test_mask.numpy())
		# np.save(f'val_mask_{seed}', val_mask.numpy())

		# train_mask = torch.from_numpy(np.load('train_mask_42.npy', allow_pickle=True))
		# val_mask = torch.from_numpy(np.load('val_mask_42.npy', allow_pickle=True))
		# test_mask = torch.from_numpy(np.load('test_mask_42.npy', allow_pickle=True))

		train_mask = dataset['author'].train_mask
		val_mask = dataset['author'].val_mask
		test_mask = dataset['author'].test_mask
		#torch.save({'feature':x,'graph_paper':data_APA,'graph_term': data_APTPA, 'graph_c': data_APCPA, 'labels':y, 'train':train_mask,'val':val_mask,'test':test_mask}, 'combined_author.pt')
		# print(torch.sum(train_mask==True))
		# print(torch.sum(val_mask==True))
		# print(torch.sum(test_mask==True))

	return dataset, data_APA, data_APCPA, data_APTPA, train_mask, val_mask, test_mask
def load_IMDB(seed=42,path=''):
	if path != '':
		os.chdir(path)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
	    torch.cuda.manual_seed(seed)
	metapaths = [[('movie', 'actor'), ('actor', 'movie')],
	         [('movie', 'director'), ('director', 'movie')],
	         [('movie', 'actor'), ('actor', 'movie'),('movie', 'director'), ('director', 'movie')],
	         [('movie', 'director'), ('director', 'movie'),('movie', 'actor'), ('actor', 'movie')]]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
	                       drop_unconnected_node_types=True)

	dataset = IMDB(root='tmp/IMDB',transform=transform)[0]
	#print(dataset)
	dataset2 = IMDB(root='tmp/IMDB2')[0]
	#print(dataset2)
	x = dataset2['movie'].x
	#print("x",x.shape)
	y = dataset2['movie'].y

	num_movies = 4278
	num_directors = 2081
	num_actors = 5257

	# MD_edge = dataset2['movie','to','director'].edge_index
	# DM_edge = dataset2['director','to','movie'].edge_index
	# movie_to_director = torch.zeros((num_movies, num_directors))
	# movie_to_director[MD_edge[0], MD_edge[1]] = 1
	# director_to_movie = torch.zeros((num_directors, num_movies))
	# director_to_movie[DM_edge[0], DM_edge[1]] = 1
	# MDM = torch.matmul(movie_to_director, director_to_movie)
	# MDM = (MDM > 0).int()

	# MA_edge = dataset2['movie','to','actor'].edge_index
	# AM_edge = dataset2['actor','to','movie'].edge_index
	# movie_to_actor = torch.zeros((num_movies, num_actors))
	# movie_to_actor[MA_edge[0], MA_edge[1]] = 1
	# actor_to_movie = torch.zeros((num_actors, num_movies))
	# actor_to_movie[AM_edge[0], AM_edge[1]] = 1
	# MAM = torch.matmul(movie_to_actor, actor_to_movie)
	# MAM = (MAM > 0).int()

	# MAMDM = torch.matmul(MAM, MDM)
	# MAMDM = (MAMDM > 0).int()

	# MDMAM = torch.matmul(MDM, MAM)
	# MDMAM = (MDMAM > 0).int()

	# torch.save({'feature':x,'graph_actor':MAM,'graph_director': MDM, 'actor': MAMDM, 'director':MDMAM, 'labels':y}, 'combined_movie.pt')

	MAM = torch.load('combined_movie.pt')['graph_actor']
	MDM = torch.load('combined_movie.pt')['graph_director']	
	AMA = torch.load('combined_movie.pt')['actor']
	DMD = torch.load('combined_movie.pt')['director']

	data_MAM = Data(x=x, edge_index=dense_to_sparse(MAM.fill_diagonal_(1))[0], y=y)
	data_MDM = Data(x=x, edge_index=dense_to_sparse(MDM.fill_diagonal_(1))[0], y=y)
	data_AMA = Data(x=x, edge_index=dense_to_sparse(AMA.fill_diagonal_(1))[0], y=y)
	data_DMD = Data(x=x, edge_index=dense_to_sparse(DMD.fill_diagonal_(1))[0], y=y)
	#degrees = degree(data_MAM.edge_index[0], num_nodes=num_movies)
	# print(f'node{num_movies}')
	# print(f'degree:{(degrees <= 1).sum().item()}')

	train_mask = dataset['movie'].train_mask
	val_mask = dataset['movie'].val_mask
	test_mask = dataset['movie'].test_mask

	# print(torch.sum(train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))

	return dataset, data_MAM, data_MDM, data_AMA, data_DMD, train_mask, val_mask, test_mask


def load_ACM(seed=42, path=''):
	if path != '':
		os.chdir(path)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
	    torch.cuda.manual_seed(seed)
	metapaths = [[('paper', 'author'), ('author', 'paper')],
	         [('paper', 'subject'), ('subject', 'paper')],]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
	                       drop_unconnected_node_types=True)

	dataset = HGBDataset(root='tmp/ACM',name='ACM',transform=transform)[0]
	dataset2 = HGBDataset(root='tmp/ACM2',name='ACM')[0]
	x = dataset2['paper'].x
	#print("x",x.shape)
	y = dataset2['paper'].y
	num_papers = 3025
	num_authors = 5959
	num_subjects = 56


	#print(dataset2)
	# PA_edge = dataset2['paper','to','author'].edge_index
	# AP_edge = dataset2['author','to','paper'].edge_index
	# paper_to_author = torch.zeros((num_papers, num_authors))
	# paper_to_author[PA_edge[0], PA_edge[1]] = 1
	# author_to_paper = torch.zeros((num_authors, num_papers))
	# author_to_paper[AP_edge[0], AP_edge[1]] = 1
	# PAP = torch.matmul(paper_to_author, author_to_paper)
	# PAP = (PAP > 0).int()

	# PS_edge = dataset2['paper','to','subject'].edge_index
	# SP_edge = dataset2['subject','to','paper'].edge_index
	# paper_to_subject = torch.zeros((num_papers, num_subjects))
	# paper_to_subject[PS_edge[0], PS_edge[1]] = 1
	# subject_to_paper = torch.zeros((num_subjects, num_papers))
	# subject_to_paper[SP_edge[0], SP_edge[1]] = 1
	# PSP = torch.matmul(paper_to_subject, subject_to_paper)
	# PSP = (PSP > 0).int()

	# torch.save({'feature':x,'graph_author':PAP,'graph_subject': PSP, 'labels':y}, 'combined_paper.pt')

	PAP = torch.load('combined_paper.pt')['graph_author']
	PSP = torch.load('combined_paper.pt')['graph_subject']

	data_PAP = Data(x=x, edge_index=dense_to_sparse(PAP.fill_diagonal_(1))[0], y=y)
	data_PSP = Data(x=x, edge_index=dense_to_sparse(PSP.fill_diagonal_(1))[0], y=y)

	train_mask = dataset['paper'].train_mask
	test_mask = dataset['paper'].test_mask

	train_indices = train_mask.nonzero(as_tuple=True)[0]
	val_size = int(0.3 * len(train_indices))
	val_indices = train_indices[:val_size]
	new_train_indices = train_indices[val_size:]

	val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
	val_mask[val_indices] = True

	new_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
	new_train_mask[new_train_indices] = True

	# print(torch.sum(new_train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))
	#torch.save({'feature':x,'graph_author':data_PAP,'graph_subject': data_PSP, 'labels':y,'train':new_train_mask,'val':val_mask,'test':test_mask}, 'paper.pt')
	return dataset, data_PAP, data_PSP, new_train_mask, val_mask, test_mask

def load_Urban(seed=42, path=''):
	if path != '':
		os.chdir(path)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	data = torch.load("urban/urban.pt")
	# print(data)
	# print(torch.min((data[("0","to","1")]["edge_index"][0])))
	# print(torch.min((data[("0","to","2")]["edge_index"][0])))
	# print(torch.min((data[("0","to","3")]["edge_index"][0])))
	x = data['0'].x

	y = torch.argmax(data['0'].y,dim=1)
	#print(y.shape)
	# print(len(torch.unique(y)))
	# print(torch.max(y))
	# print(torch.min(y))

	num_zeros = data['0'].x.size(0)

	zo_edge = data['0','to','1'].edge_index
	#print(zo_edge[1].max().item() + 1)
	zo_adj = to_dense_adj(zo_edge)[0]
	# ZOZ = torch.matmul(zo_adj,zo_adj.T)
	# ZOZ = (ZOZ > 0).int()

	zt_edge = data['0','to','2'].edge_index
	#print(zt_edge[1].max().item() + 1)
	zt_adj = to_dense_adj(zt_edge)[0]
	# ZTZ = torch.matmul(zt_adj,zt_adj.T)
	# ZTZ = (ZTZ > 0).int()

	zth_edge = data['0','to','3'].edge_index
	#print(zth_edge[1].max().item() + 1)
	zth_adj = to_dense_adj(zth_edge)[0]
	# ZThZ = torch.matmul(zth_adj,zth_adj.T)
	# ZThZ = (ZThZ > 0).int()

	# torch.save({'feature':x, 'graph_one':ZOZ, 'graph_two':ZTZ,'graph_three': ZThZ, 'labels':y}, 'combined_urban.pt')
	metapaths = [[('0','1'),('1','0')],
				[('0','2'),('2','0')],
				[('0','3'),('3','0')]]

	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
	                       drop_unconnected_node_types=True)
	data['0'].y = y
	# print(data)
	# print(x.shape)

	dataset = transform(data)
	ZOZ = torch.load('combined_urban.pt')['graph_one']
	ZTZ = torch.load('combined_urban.pt')['graph_two']
	ZThZ = torch.load('combined_urban.pt')['graph_three']

	data_ZOZ = Data(x=x, edge_index=dense_to_sparse(ZOZ)[0], y=y)
	data_ZTZ = Data(x=x, edge_index=dense_to_sparse(ZTZ)[0], y=y)
	data_ZThZ = Data(x=x, edge_index=dense_to_sparse(ZThZ)[0], y=y)
	# num_nodes = num_zeros
	# float_mask = np.random.permutation(np.linspace(0, 1, num_nodes))
	# #train_idx = np.where(float_mask <= 0.01)[0]
	# #val_idx = np.where((float_mask > 0.01) & (float_mask <= 0.02))[0]
	# #test_idx = np.where(float_mask > 0.02)[0]
	# train_idx = np.where(float_mask <= 0.5)[0]
	# val_idx = np.where((float_mask > 0.5) & (float_mask <= 0.8))[0]
	# test_idx = np.where(float_mask > 0.8)[0]
	# train_mask = get_binary_mask(num_nodes, train_idx)
	# val_mask = get_binary_mask(num_nodes, val_idx)
	# test_mask = get_binary_mask(num_nodes, test_idx)

	# np.save("train_mask_urban.npy", train_mask)
	# np.save("val_mask_urban.npy", val_mask)
	# np.save("test_mask_urban.npy", test_mask)
	train_mask = torch.from_numpy(np.load('train_mask_urban.npy', allow_pickle=True))
	val_mask = torch.from_numpy(np.load('val_mask_urban.npy', allow_pickle=True))
	test_mask = torch.from_numpy(np.load('test_mask_urban.npy', allow_pickle=True))

	# print(torch.sum(train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))
	return dataset,data_ZOZ,data_ZTZ,data_ZThZ,train_mask,val_mask,test_mask

def load_company(seed=42,dataname='a',path=''):
	if path != '':
		os.chdir(path)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
	    torch.cuda.manual_seed(seed)
	metapaths = [[('company', 'organization'), ('organization', 'company')],
             [('company', 'person'),('person','company')]]
	transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                           drop_unconnected_node_types=True)

	X = torch.from_numpy(np.load('company_data_new.npy',allow_pickle=True).item()['feature'])
	CPC = torch.from_numpy(np.load('company_data_new.npy',allow_pickle=True).item()['graph_people'])
	COC = torch.from_numpy(np.load('company_data_new.npy',allow_pickle=True).item()['graph_org'])
	CO = torch.from_numpy(np.load('CO.npy'))
	CP = torch.from_numpy(np.load('CP.npy'))
	Y = torch.from_numpy(np.load('company_data_new.npy',allow_pickle=True).item()['labels'])
	data = HeteroData()
	data['company'].x = X
	data['company'].y = Y

	data['company', 'organization'].edge_index = dense_to_sparse(CO.fill_diagonal_(1))[0]
	data['company', 'person'].edge_index = dense_to_sparse(CP.fill_diagonal_(1))[0]
	data['organization', 'company'].edge_index = data['company', 'organization'].edge_index[[1, 0]]
	data['person', 'company'].edge_index = data['company', 'person'].edge_index[[1, 0]]

	dataset = transform(data)
	#print(dataset)
	#print(data)

	data_CPC = Data(x=X, edge_index=dense_to_sparse(CPC.fill_diagonal_(1))[0], y=Y)
	data_COC = Data(x=X, edge_index=dense_to_sparse(COC.fill_diagonal_(1))[0], y=Y)

	train_mask = torch.from_numpy(np.load('train_mask_42.npy', allow_pickle=True))
	val_mask = torch.from_numpy(np.load('val_mask_42.npy', allow_pickle=True))
	test_mask = torch.from_numpy(np.load('test_mask_42.npy', allow_pickle=True))

	# print(torch.sum(train_mask==True))
	# print(torch.sum(val_mask==True))
	# print(torch.sum(test_mask==True))


	return dataset, data_CPC, data_COC, train_mask, val_mask, test_mask

#load_ACM(seed=42,path='')
#load_DBLP(seed=42,dataname='DBLP',path='')
#load_Urban(seed=42,path='')
#load_IMDB(seed=42,path='')
#load_company(seed=42,dataname='',path='')

