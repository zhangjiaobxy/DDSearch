#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# *********************************************************************
# Usage example:
# 				python DDSearch.py --topk 10 --label DS3 --beta 0.5
# *********************************************************************


# import the modules needed to run the script
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import os, sys, subprocess, shutil, time, math, random
import argparse
import numpy as np
import networkx as nx
import pandas as pd
import logging
import gensim.models as gm
import codecs
from sklearn.ensemble import AdaBoostRegressor  
from sklearn.tree import DecisionTreeRegressor


# create folders
rawDBDir = './../data/rawData/db/' # raw data of database networks
rawQueryDir = './../data/rawData/query/' # raw data of query set networks
randomWalksDir = './../data/randomWalks/' # random walks for each network
doc2vecDir = './../data/doc2vec/' # document embedding
naDBDir = './../data/NA/db/' # network format for the NA (network alignment)
naQueryDir = './../data/NA/query/'
csvDir = './../data/feaLabel/' # csv file for the kerasRegression

if not os.path.exists(randomWalksDir):
	os.makedirs(randomWalksDir)

if not os.path.exists(doc2vecDir):
	os.makedirs(doc2vecDir)

if not os.path.exists(naDBDir):
	os.makedirs(naDBDir)

if not os.path.exists(naQueryDir):
	os.makedirs(naQueryDir)

if not os.path.exists(csvDir):
	os.makedirs(csvDir)


# define parameters
def parse_args():

	parser = argparse.ArgumentParser(description="Run DDSearch")

	parser.add_argument('--rawDBDir', nargs='?', default= rawDBDir,
                    help='Input db graph directory')

	parser.add_argument('--rawQueryDir', nargs='?', default= rawQueryDir,
	                    help='Input query graph directory')

	parser.add_argument('--randomWalks', nargs='?', default= randomWalksDir + 'randomWalks.txt',
                    	help='Output random walks directory')

	parser.add_argument('--d', type=int, default=30,
	                    help='Number of feature dimensions. Default is 30.')

	parser.add_argument('--l', type=int, default=15,
	                    help='Length of walk per vertex. Default is 15.')  

	parser.add_argument('--r', type=int, default=1,
	                    help='Number of random walks per vertex. Default is 1.')  

	parser.add_argument('--c', type=int, default=6,
                    	help='Context (window) size for optimization. Default is 6.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=4,
	                    help='Number of parallel workers. Default is 4.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter (BFS controller). Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter (DFS controller). Default is 1.')

	parser.add_argument('--beta', type=float, default=0.5,
	                    help='Ratio parameter to control weight of Time and BLAST score. Default is 0.5.')

	parser.add_argument('--topk', type=int, default=10, 
		help='Output topk similar graphs. Default topk is 10.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=True)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	parser.add_argument('--label', nargs='?', default= 'DS3',
		help='NA label could be DS3, ICS, EC. Default is DS3.')

	return parser.parse_args()


class Graph():
	def __init__(self, snapshot, is_directed, p, q):
		self.G = snapshot
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def conduct_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		for walk_iter in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.conduct_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}


		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	or: http://shomy.top/2017/05/09/alias-method-sampling/
	for details
	'''
	
	'''
	probs: 某个概率分布
	return: Alias数组与Prob数组
	'''

	K = len(probs)
	q = np.zeros(K) # 对应Prob数组
	J = np.zeros(K, dtype=np.int) # 对应Alias数组

	# Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.

	smaller = [] # 存储比1小的列
	larger = [] # 存储比1大的列
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob # probabilty
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	# Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.

    # 通过拼凑，将各个类别都凑为1
	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large # 填充Alias数组
	    q[large] = q[large] - (1.0 - q[small]) # 将大的分到小的上
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q


def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''

	'''
	input: Prob数组和Alias数组
	output: 一次采样结果
	'''

	K = len(J)

	# Draw from the overall uniform mixture
	kk = int(np.floor(np.random.rand()*K)) # 随机取一列

	# Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]


# read input graph
def read_graph(args, inputFile):

	inFile = open(inputFile, 'r') # read raw data
	
	lines = inFile.readlines()

	unique_times = sorted(set(float(line.split('\t')[2].split('|')[0]) for line in lines)) # unique start_time

	snapshots = []

	for unique_time in unique_times:

		snapshot = [line for line in lines if line.split('\t')[2].split('|')[0] == str(unique_time)]

		g = nx.parse_edgelist(snapshot, nodetype=int, delimiter='\t', data=(('weight', str),), create_using=nx.DiGraph())
		# https://networkx.github.io/documentation/stable/reference/readwrite/generated/networkx.readwrite.edgelist.parse_edgelist.html#networkx.readwrite.edgelist.parse_edgelist

		# weight = beta * weightedTime + (1-beta) * blastScore
		if args.weighted: # weightedTime=time_end - time_start
			for edge in g.edges(data=True):

				edge[2]['weight'] = args.beta * abs(float(edge[2]['weight'].strip().split('|')[1]) - float(edge[2]['weight'].strip().split('|')[0])) +\
								(1-args.beta) * float(edge[2]['weight'].strip().split('|')[2])

		else: # unweightedTime = 1
			for edge in g.edges(data=True):

				edge[2]['weight'] = args.beta * 1 + (1-args.beta) * float(edge[2]['weight'].strip().split('|')[2])
		
		if not args.directed:
			g = g.to_undirected()

		snapshots.append(g)

	inFile.close()
	return snapshots


# for a dynamic graph, get all the walks for all the snapshots
def get_walks(args):

	outWalks = open(args.randomWalks, 'w') # write the walks for each input dynamic graph

	fl_db = [args.rawDBDir + f for f in os.listdir(args.rawDBDir) if f.endswith('.txt')] # get db file names

	fl_query = [args.rawQueryDir + f for f in os.listdir(args.rawQueryDir) if f.endswith('.txt')] # get query file names
	fl = fl_db + fl_query # get all the file names

	num_db = len(fl_db) # db number
	num_query = len(fl_query) # query number

	if '-' in fl[0]:
		fl_sort = sorted(fl, key=lambda x: int(x.strip().split('/')[-1].split('-')[0])) # sort file name in ascending order
	else:
		fl_sort = sorted(fl, key=lambda x: int(x.strip().split('/')[-1].split('.')[0])) # sort file name in ascending order

	all_walks = [] # all the walks for all the snapshots of all the files in the inputDir
	for f in fl_sort:
		snapshots = read_graph(args, f) # snapshots is the dynamic graph at several time points
		for snapshot in snapshots: # each snapshot is a static graph at a certain time point
			G = Graph(snapshot, args.directed, args.p, args.q)
			G.preprocess_transition_probs()
			walks_snapshot = G.simulate_walks(args.r, args.l) # all walks for each snapshot
			for each_walk in walks_snapshot:
				for each_node in each_walk:
					outWalks.write(str(each_node) + ' ')
					all_walks.append(str(each_node) + ' ')
				outWalks.write('. ')
				all_walks.append('. ')
		outWalks.write('\n') # the walks of each dynamic graph is separated by '\n'
		all_walks.append('\n')
	outWalks.close()

	return num_db, num_query, fl_db, fl_query, all_walks


def doc_embedding():
	# ----------------------- train ----------------------------
	seed = 23
	min_count = 1  # ignore all words with total frequency lower than this
	sampling_threshold = 1e-5  #  threshold for configuring which higher-frequency words are randomly downsampled
	negative_size = 5  # negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20).
	train_epoch = 20  # number of iterations (epochs) over the corpus. The default inherited from Word2Vec is 5, but values of 10 or 20 are common in published ‘Paragraph Vector’ experiments
	dm = 0 #0 = dbow; 1 = dmpv
	corpus = args.randomWalks # input corpus, used for train and infer
	saved_model = doc2vecDir + 'doc2vecTrained.bin' # output model

	#enable logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	#train doc2vec model
	docs = gm.doc2vec.TaggedLineDocument(corpus)

	# mode without pretrained_emb
	model = gm.Doc2Vec(docs, vector_size=args.d, window=args.c, seed = seed, min_count=min_count, \
		sample=sampling_threshold, workers=args.workers, hs=0, dm=dm, negative=negative_size, dbow_words=1, \
		dm_concat=1, epochs=train_epoch)

	#save model
	model.save(saved_model)

	# ----------------------- infer ----------------------------
	# inference hyper-parameters
	start_alpha=0.01
	infer_epoch=1000
	infered_file = doc2vecDir + 'doc2vecInfered.txt' # output infered vector

	# load_model = g.Doc2Vec.load(model) # load model, if it is pre-saved in a path
	corpus = [ x.strip().split() for x in codecs.open(corpus, "r", "utf-8").readlines() ]

	# infer vectors
	vectors = []
	infered_output = open(infered_file, "w")
	i = 0 # count the number of files
	for doc in corpus:
		i += 1
		vector = str(i) + ' ' + " ".join([str(x) for x in model.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)]) + "\n"
		infered_output.write(vector)
		vectors.append(vector)
	infered_output.flush()
	infered_output.close()
	vectors = sorted(vectors, key=lambda x: int(x.split(' ')[0]))

	return infered_file


def get_label(args, fl_db, fl_query):
	fl = fl_db + fl_query
	for f in fl:
		if f in fl_db:
			outDirTemp = naDBDir
		elif f in fl_query:
			outDirTemp = naQueryDir
		inFile = open(f, 'r') # read DDSearch format file
		outFile = open(outDirTemp + f.split('/')[-1], 'w') # generate dynamagna++ format file
		ls = inFile.readlines()
		for l in ls:
			l_spl = l.strip().split('\t')
			l_new = l_spl[2].split('|')[0] + '\t' + l_spl[2].split('|')[1] + '\t' + l_spl[0] + '\t' + l_spl[1] + '\n'
			outFile.write(l_new)
		inFile.close()
		outFile.close()


	''' ------------------- command line of dynamagna++ -------------------
	./dynamagna -G ./../data/rawData/query/241-1984-06.txt -H ./../data/rawData/db/1-1996-08.txt -o ./test_run_new -m DS3 -p 10 -n 10 -t 8
	To optimize DS^3, pass `DS3' to the -m flag, (dynamic)
	To optimize EC, pass `EC' to the -m flag, (static)
	to optimize ICS, pass `ICS', (static)
	to optimize S^3, pass `S3'.	(static)
	'''

	cmd_temp = './dynamagna -G [filename_query] -H [filename_db] -o [f_out] -m [label] -p 10 -n 10 >> screenDynamagnaOut'
	if args.label == 'DS3':
		naOutDir = './../data/NA/out/DS3/' # output
	elif args.label == 'S3':
		naOutDir = './../data/NA/out/S3/'
	elif args.label == 'EC':
		naOutDir = './../data/NA/out/EC/'
	elif args.label == 'ICS':
		naOutDir = './../data/NA/out/ICS/'
	alnDir = naOutDir + 'aln/' # alignment files
	scoreDir = naOutDir + 'score/' # label values

	if not os.path.exists(naOutDir):
		os.makedirs(naOutDir)
	if not os.path.exists(alnDir):
		os.makedirs(alnDir)
	if not os.path.exists(scoreDir):
		os.makedirs(scoreDir)
	get_label_start = float(round(time.time() * 1000))  # millisecond

	for f_query in fl_query:
		f_G = f_query.strip().split('/')[-1]
		for f_db in fl_db:
			f_H = f_db.strip().split('/')[-1]
			f_O = naOutDir + f_G.replace('.txt', '_') + f_H.replace('.txt', '_')
			if args.label == 'ICS': # since we get errors when using ICS as the optimum metric, we use EC as the optimum metric to run dynamagna and get ICS
				naMetric = 'EC'
			else:
				naMetric = args.label
			cmd = cmd_temp.replace('[filename_query]', naQueryDir + f_G).replace('[filename_db]', naDBDir + f_H).\
			replace('[label]', naMetric).replace('[f_out]', f_O)

			os.system(cmd)


	get_label_end = float(round(time.time() * 1000))  # millisecond
	get_label_time = get_label_end - get_label_start

	try:
		os.remove('screenDynamagnaOut')
	except OSError:
		pass


	# get label
	for f in os.listdir(naOutDir):
		if f.endswith('.sif'):
			os.remove(naOutDir + f)
		elif f.endswith('_alignment.txt'):
			shutil.move(naOutDir + f, alnDir + f)
		elif f.endswith('_stats.txt'):
			shutil.move(naOutDir + f, scoreDir + f)

	labelDir = './../data/NA/out/label/'
	if not os.path.exists(labelDir):
		os.makedirs(labelDir)
	label_file = labelDir + 'dynamagna_' + args.label + '_label.txt'
	fname_score = open(label_file, 'w')
	lname_scores = []
	for f in os.listdir(scoreDir):
		if f.endswith('.txt'):
			inFile = open(scoreDir + f, 'r')
			ls = inFile.readlines()
			for l in ls:
				if l.startswith(args.label + '_score '):
					score = l.strip().split(' ')[1]
					lname_score = f.split('_')[0].split('-')[0] + '\t' + f.split('_')[1].split('-')[0] + '\t' + score + '\n'
					lname_scores.append(lname_score)
			
	lname_scores = sorted(lname_scores, key=lambda x: int(x.split('\t')[1])) # sort by db file 
	lname_scores = sorted(lname_scores, key=lambda x: int(x.split('\t')[0])) # sort by query file
	for lname_score in lname_scores:
		fname_score.write(lname_score)

	return label_file, get_label_time # the format is: query_file_name	db_file_name	label


def get_feaLable(args, num_db, num_query,infered_file, label_file):

	csv_feaLabel = csvDir + 'dynamagna_fea_' + args.label + '.csv'
	outFile = open(csv_feaLabel, 'w')
	outFile.write('q_name'+','+'db_name'+ ',' + ','.join(['q'+str(x) for x in range(1,args.d+1)]) + ',' +\
		','.join(['db'+str(y) for y in range(1, args.d+1)]) + ',' + 'dynamagna_'+args.label+'_label' +'\n')

	# read feature
	in_fea = open(infered_file, 'r')
	ls_fea = in_fea.readlines()
	vec_db = ls_fea[0:num_db]
	vec_q = ls_fea[num_db:num_db+num_query]

	# read label
	in_lab = open(label_file, 'r')
	ls_lab = in_lab.readlines()

	# feaLabels = []
	for label in ls_lab:
		l_spl = label.strip().split('\t')
		for v_q in vec_q:
			v_q_spl = v_q.strip().split(' ')
			for v_db in vec_db:
				v_db_spl = v_db.strip().split(' ')
				if l_spl[0] == v_q_spl[0] and l_spl[1] == v_db_spl[0]:
					feaLabel = l_spl[0] + ',' + l_spl[1] +','+ \
					','.join([q for q in v_q_spl[1:]]) + ',' + \
					','.join([db for db in v_db_spl[1:]]) + ',' + l_spl[2] + '\n'
					outFile.write(feaLabel)
	outFile.close()

	return csv_feaLabel
				

# define the base model of keras
def keras_model():

	# create model
	kerasModel = Sequential()
	kerasModel.add(Dense(13, input_dim=60, kernel_initializer='normal', activation='relu')) # input_dim: the input variables
	kerasModel.add(Dense(6, kernel_initializer='normal', activation='relu'))
	kerasModel.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

	# Compile model
	kerasModel.compile(loss='mean_squared_error', optimizer='adam')
	# optimizer: adam, sgd, Adagrad, Adadelta, RMSprop 
	# https://faroit.github.io/keras-docs/0.2.0/optimizers/#adam
	return kerasModel


def KerasRegression(args, csv_feaLabel, get_label_time):

	########################## clean data ##################################################
	df = pd.read_csv(csv_feaLabel, skipinitialspace=True, delimiter=',')
	df = df.fillna(0)  # replace NaN with 0
	df = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]  # remove 0 columns and rows


	########################## initialization ##################################################
	k = 10  # 10-fold cross-validation
	n = df.shape[0]/k  # size of each fold
	inc = 0
	feature = df.columns[2:(df.shape[1]-1)]  # feature column names
	label = df.columns[df.shape[1]-1]  # label column name
	db_size = len(df.db_name.unique())  # database size
	q_size = len(df.q_name.unique())  # the number of query networks
	train_time_s = []
	test_time_s = []
	out_df_topk = pd.DataFrame()  # top k similar networks against each query network
	out_df_save = pd.DataFrame()  # all the networks


	########################## k-fold cross-validation ##########################################
	for i in range(0,k):
		s1 = int(i * n)
		s2 = int((i+1) * n)
		test = df[s1:s2]  # test data
		train = (df[:s1]).append(df[s2:])  # train data

		########################## model: train / test ###############################################
		# train
		train_start = float(round(time.time() * 1000))  # millisecond
		keras_train = KerasRegressor(build_fn=keras_model, epochs=100, batch_size=40, verbose=0)
		keras_train.fit(train[feature], train[label])
		train_end = float(round(time.time() * 1000))
		train_time = train_end - train_start
		train_time_s.append(train_time)

		# test
		test_start = float(round(time.time() * 1000))
		keras_prediction = keras_train.predict(test[feature])
		test_end = float(round(time.time() * 1000))
		test_time = test_end - test_start
		test_time_s.append(test_time)

		predictions = pd.DataFrame(keras_prediction, columns=['predictions'])  # prediction column
		q_db_df = pd.concat([test['q_name'],test['db_name'],test[label]],axis=1).reset_index()  # q_name, db_name, and label dataframe
		out_df = (pd.concat([q_db_df, predictions],axis=1))  # q_name, db_name, label, and prediction dataframe
		out_df_save = out_df_save.append(out_df)


	########################## avg time cost ##########################################
	avg_train_time = np.mean(train_time_s)  # average training time for each fold (millisecond)
	avg_test_time = (np.mean(test_time_s))/(q_size/k)  # average query time for each query network


	######################### output / write to file ####################################################
	topkDir = './../data/output/topk/'
	allDir = './../data/output/allResult/'
	terminalDir = './../data/output/terminalOut/'
	if not os.path.exists(topkDir):  # the topk networks
	    os.makedirs(topkDir)
	if not os.path.exists(allDir):  # all the networks
	    os.makedirs(allDir)
	if not os.path.exists(terminalDir): # terminal output
		os.makedirs(terminalDir)


	topk = args.topk	
	# save topk predictions
	topk_perc = int(math.ceil(topk/100 * db_size))  # the number of the topk networks
	q_test_size = len(out_df_save.q_name.unique())  # the number of query networks 
	for q in range(0,q_test_size):  # get top k similar networks against each query network
		out_df_e = (out_df_save[inc:inc+db_size]).sort_values(['predictions'],ascending=False)[:topk_perc]
		inc = inc + db_size
		out_df_topk = out_df_topk.append(out_df_e)
	out_df_topk = out_df_topk.sort_values(['q_name'],ascending=True)
	header = ['q_name','db_name',label,'predictions']
	out_df_topk.to_csv(topkDir+ csv_feaLabel.strip().split('/')[-1].replace('.csv','_top'+str(topk)+'%.csv'),\
	header=True, columns=header, index = False)  # top k similar network output file

	########################## simpson overlap ##########################################
	# save all predictions
	out_df_save = out_df_save.sort_values(['q_name'],ascending=True)
	out_df_save.to_csv(allDir+ csv_feaLabel.strip().split('/')[-1].replace('.csv', '_all.csv'), \
		header=True, columns=header, index = False)

	ol_f = open(allDir+ csv_feaLabel.strip().split('/')[-1].replace('.csv', '_all.csv'), 'r')
	ols = ol_f.readlines()[1:]
	qname_set = set()
	for ol in ols:
		qname = ol.strip().split(',')[0]
		qname_set.add(float(qname))  # all the query network name
	ol_values = []
	for e_qname in qname_set:
		block = []  # all db network related to each of the query network
		for ol in ols:
			if float(ol.strip().split(',')[0]) == e_qname:
				block.append(ol)

		block_label = sorted(block,key=lambda x:float(x.strip().split(',')[2]), reverse=True)  # sort each query block by the label column
		block_pred = sorted(block,key=lambda x:float(x.strip().split(',')[3]), reverse=True)  # sort each query block by the prediction column

		db_labels = []  # db names column by sorting labels in descending order
		db_preds = []  # db names column by sorting predictions in descending order
		for e_block_label in block_label:
			db_label = e_block_label.strip().split(',')[1]
			db_labels.append(db_label)
		db_labels_topk=db_labels[:topk_perc]
		for e_block_pred in block_pred:
			db_pred = e_block_pred.strip().split(',')[1]
			db_preds.append(db_pred)
		db_preds_topk=db_preds[:topk_perc]
		ol_value = len(set(db_labels_topk).intersection(db_preds_topk))/topk_perc
		ol_values.append(ol_value)
	ol = round(np.mean(ol_values),5)  # save 5 digit number


	if os.path.exists(allDir):  # remove the dir and its files
		shutil.rmtree(allDir)

	
	print('\n*********************************************************************\n')	
	print('The output file is: ', csv_feaLabel.strip().split('/')[-1], '\n')
	print('Simpson overlap coefficient for top ' + str(topk) + ' is: ', str(ol), '\n')
	print('Average training time for',k,'fold cross validation is: ',avg_train_time, 'msec.\n')
	print('Average testing time for',k,'fold cross validation is: ',avg_test_time, 'msec.\n')
	print('Run dynamagna and get label time is: ', get_label_time, '\n')
	print('*********************************************************************\n')

	terminalOut	= open(terminalDir + csv_feaLabel.strip().split('/')[-1].replace('.csv', '_top'+ str(topk) +'%_terminalOut.csv'),'w')
	terminalOut.write('*********************************************************************\n\n')
	terminalOut.write('The output file is: '+ csv_feaLabel.strip().split('/')[-1] + '\n')
	terminalOut.write('Simpson overlap coefficient for top ' + str(topk) + '%'+' is: '+str(ol)+ '\n')
	terminalOut.write('Average training time for'+str(k)+'fold cross validation is: '+str(avg_train_time)+ 'msec.\n' + \
		'Average testing time for'+str(k)+'fold cross validation is: '+str(avg_test_time)+ 'msec.\n' + \
		'Run dynamagna and get label time is: '+str(get_label_time)+ 'msec.\n' + \
		'\n*********************************************************************\n')

	

# main
def main(args):

	num_db, num_query, fl_db, fl_query, all_walks = get_walks(args) # all_walks: all walks for all the dynamic graphs

	infered_file = doc_embedding() # each dynamic graph can be represented as a vector
 
	label_file, get_label_time = get_label(args, fl_db, fl_query) # run dynamagna and get label

	csv_feaLabel = get_feaLable(args, num_db, num_query, infered_file, label_file) # get a csv file of feature and label

	KerasRegression(args, csv_feaLabel, get_label_time) # run AdaBoost regression



if __name__ == '__main__':
	args = parse_args()
	main(args)


print('Done.')




