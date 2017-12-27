import numpy as np
import math as maths
import random as rand
import datetime
import scipy.cluster.vq as vq


def compute_design_matrix(X, centers, spreads):
	#Generates design matrix 
	# use broadcast
	basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads) * (X - centers), axis=2) / (-2)).T
	# insert ones to the 1st col
	return np.insert(basis_func_outputs, 0, 1, axis=1)

def closed_form_sol(L2_lambda, design_matrix, output_data):
	#Closed form solution provided by TA's and used for the project
	#takes input: lambda, design matrix, target data
	#returns closed form solution for input data
	return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix),
							np.matmul(design_matrix.T, output_data)
							).flatten()

def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data):
	#SGD algorithm explained and given by Class TA
	N, D = design_matrix.shape
	weights = np.zeros([1, D])
	for epoch in range(num_epochs):
		for i in range(maths.ceil(N / minibatch_size)):
			lower_bound = i * minibatch_size
			upper_bound = min((i+1)*minibatch_size, N)
			Phi = design_matrix[lower_bound : upper_bound, :]
			t = output_data[lower_bound : upper_bound, :]
			E_D = np.matmul( (np.matmul(Phi, weights.T)-t).T, Phi)
			E = (E_D + L2_lambda * weights) / minibatch_size
			weights = weights - learning_rate * E
	return weights.flatten()

def SGD_sol2(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data, validation_output_data, validation_data, p, val_size ):
	#used the Gradient solution method explained by Class TA's and combined it with Early Stop algorithm
	#takes input: ita (rate of learning), batch size, number of iterations, lambda, design matrix, target data, validation dataset and validation output set
	#returns the weight, validation error , and number of iterations used to train weight
	weights = np.zeros([1, D])
	theta = np.zeros([1, design_matrix.shape[1]])
	n = 10
	i2 = 0
	j = 0
	v = float("inf")
	v_dash = v
	theta_s = theta
	i_s = i2

	for epoch in range(num_epochs):
		for i in range(maths.ceil(N / minibatch_size)):
			lower_bound = i * minibatch_size
			upper_bound = min((i+1)*minibatch_size, N)
			Phi = design_matrix[lower_bound : upper_bound, :]
			t = output_data[lower_bound : upper_bound, :]
			E_D = np.matmul( (np.matmul(Phi, weights.T)-t).T, Phi)
			E = (E_D + L2_lambda * weights) / minibatch_size
			weights = weights - learning_rate * E
		
		if epoch == i2 :
			if  j < p :
				theta = weights.flatten()
				i2 = i2 + n       #which is the no. of steps
				v_dash =  err_rms(t=validation_output_data, phi=validation_data, weights=theta, L2_lambda=L2_lambda, N=val_size )
				
				if  np.round(v_dash, 4) < np.round(v, 4) :
					j = 0
					theta_s = theta
					i_s = i2
					v = v_dash
				else :
					j += 1
			else :
				break
				
	return theta_s, v, i_s
	
def err_func(t, phi, weights) :
	#takes target, phi (design matrix) , and weight obtained from training as input
	# returns sum-of-squares error
	E_D = 0
	for i, xi in enumerate(phi):
		E_D += (t[i]-np.matmul(weights.T,phi[i]))**2
		#print(t[i],np.matmul(weights.T,phi[i]),(t[i]-np.matmul(weights.T,phi[i]))**2, E_D)
	return E_D/2

def err_rms(t, phi, weights, L2_lambda,N):
	#takes target, phi (design matrix) , weight obtained from training, and lambda to calculate Root mean square error
	#returns RMS error
	#uses err_func for initial calculations
	E_D = err_func(t, phi, weights)	
	E = E_D + (L2_lambda*np.matmul(weights.T,weights))/2	
	return np.sqrt((2*E)/N)

def find_center(data, cluster_nums):
	#implement K-mean algorithm to find center of cluster_nums clusters
	#slow performance due to number of loops 
	#created find_center2 for use
	centers = data[np.random.randint(data.shape[0], size=cluster_nums), :]
	clusters = np.zeros(data.size,dtype=int)
	
	#print(centers)
	#print('clusters',clusters)
	#print('cluster_arr,cluster_nums',cluster_arr,cluster_nums)
	center_old = centers+1
	
	while ( np.array_equal(center_old,centers)==False):
		#print('center_old',center_old ,'\ncenters',centers,'\n')
		cluster_arr = np.array([np.empty(data.shape)]*cluster_nums)
		cluster_index = np.zeros(cluster_nums,dtype=int)
		spreads = [np.identity(syn_training_data.shape[1])]*cluster_nums

		for i, data_point in enumerate(data):
			distance = float("inf")
			#cluster_num = None
			for j, center in enumerate(centers):
		#		print(np.linalg.norm(data_point-center))
				if(distance > np.linalg.norm(data_point-center)):
					distance = np.linalg.norm(data_point-center)
					#id cluster for this data point
					clusters[i] = j
			#np.append(clusters[cluster_num],data_point)
		# put 
		for i,data_point in enumerate(data):
			for j in range(cluster_nums):
				if clusters[i] == j:
					cluster_arr[j][cluster_index[j]] = data_point
					cluster_index[j] +=1
		#centers = np.median(clusters,axis=0)
		center_old = np.copy(centers)
		for j in range(cluster_nums):
			centers[j] = np.median(cluster_arr[j][:cluster_index[j]],axis=0)
			#print('j=',j,cluster_arr[j][:cluster_index[j]])
			spreads[j] = spreads[j]*np.cov(cluster_arr[j][:cluster_index[j]],rowvar=False)
		
		#print('new: ',centers,'\n',center_old)
		#
		#print(centers)
	return centers, spreads 

def find_center2(data, cluster_nums):
	#implement K-mean to find center of cluster_nums clusters
	#uses Scipy kmean funtion for find center and using the center get the cluster and the spread of the cluster
	centers, _ = vq.kmeans(data,cluster_nums)
	clusters = np.zeros(data.size,dtype=int)
	
	#print(centers)
	#print('clusters',clusters)
	#print('cluster_arr,cluster_nums',cluster_arr,cluster_nums)

	cluster_arr = np.array([np.empty(data.shape)]*cluster_nums)
	cluster_index = np.zeros(cluster_nums,dtype=int)
	spreads = [np.identity(data.shape[1])]*cluster_nums
	
	for i, data_point in enumerate(data):
		distance = float("inf")
		#cluster_num = None
		for j, center in enumerate(centers):
	#		print(np.linalg.norm(data_point-center))
			if(distance > np.linalg.norm(data_point-center)):
				distance = np.linalg.norm(data_point-center)
				#id cluster for this data point
				clusters[i] = j
		#np.append(clusters[cluster_num],data_point)
	# put 
	for i,data_point in enumerate(data):
		for j in range(cluster_nums):
			if clusters[i] == j:
				cluster_arr[j][cluster_index[j]] = data_point
				cluster_index[j] +=1
	#centers = np.median(clusters,axis=0)
	
	for j in range(cluster_nums):
		#print('j=',j,cluster_arr[j][:cluster_index[j]])
		spreads[j] = (spreads[j]*np.cov(cluster_arr[j][:cluster_index[j]],rowvar=False))#/10
	
	#print('new: ',centers,'\n',center_old)
	#
	#print(centers)
	return centers, spreads 

	
def early_stop(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data, validation_data,validation_output_data, p, N):
	# Early prototype of Early stop to be used with SGD_sol function but ran into loops so merged with SGD_sol and created SGD_sol2	
	theta = np.zeros([1, design_matrix.shape[1]])
	n = num_epochs
	i = 0
	j = 0
	v = float("inf")
	#theta_s = theta
	i_s = i
	
	while  j < p :
		#call SGD_sol here
		print('j=',j,v,i,theta)
		theta = SGD_sol(learning_rate, minibatch_size, i, L2_lambda, design_matrix, output_data)
		i = i + n       #which is the no. of steps
		v_dash =  err_rms(t=validation_output_data, phi=validation_data, weights=theta, L2_lambda=L2_lambda, N=N )
		if  v_dash < v :
			j = 0
			theta_s = theta
			i_s = i
			v = v_dash
		else :
			j += 1
			
	return theta_s, i_s


###################################################EXEC START############################################
print('\nstarted at:',str(datetime.datetime.now()))

L = 0.1 #lambda
patience = 10
n = 10000 #epochs

syn_input_data = np.genfromtxt('datafiles/input.csv', delimiter=',')
syn_output_data = np.genfromtxt('datafiles/output.csv', delimiter=',').reshape([-1, 1])

syn_training_data = syn_input_data[:maths.ceil(0.8*syn_input_data.shape[0])]
syn_validation_data =  syn_input_data[maths.ceil(0.8*syn_input_data.shape[0]):maths.ceil(0.8*syn_input_data.shape[0]) + maths.ceil(0.1*syn_input_data.shape[0])]
syn_test_data = syn_input_data[maths.ceil(0.9*syn_input_data.shape[0]):]

syn_training_output_data = syn_output_data[:maths.ceil(0.8*syn_input_data.shape[0])]
syn_validation_output_data =  syn_output_data[maths.ceil(0.8*syn_input_data.shape[0]):maths.ceil(0.8*syn_input_data.shape[0]) + maths.ceil(0.1*syn_input_data.shape[0])]
syn_test_output_data = syn_output_data[maths.ceil(0.9*syn_input_data.shape[0]):]


N, D = syn_training_data.shape

X = syn_training_data[np.newaxis, :, :]
X_v = syn_validation_data[np.newaxis, :, :]
X_t = syn_test_data[np.newaxis, :, :]

best_k_closed_form = 1000
best_k_sgd = 1000
min_error_closed_form = 1000
min_error_sgd = 1000
best_ita = 1

for K in [3,5,10,15,20]:
	centers, spreads = find_center2(data = syn_training_data, cluster_nums = K)
	centers = centers[:, np.newaxis, :]
	design_matrix = compute_design_matrix(X, centers, spreads)
	design_matrix_v = compute_design_matrix(X_v, centers, spreads)

	closed_soln_wt = closed_form_sol(L2_lambda=L,design_matrix=design_matrix,output_data=syn_training_output_data)
	
	print('closed_form_sol',  closed_soln_wt)
	closed_form_error = err_rms(t=syn_validation_output_data, phi=design_matrix_v, weights= closed_soln_wt, L2_lambda=L, N=syn_validation_data.shape[0])
	if min_error_closed_form > closed_form_error :
		min_error_closed_form = closed_form_error
		best_k_closed_form = K
	
	for ita in [.9, .5, .1, .05, .01] :
		print(L, patience, n, ita, K )

		weight,error, steps = SGD_sol2(learning_rate=ita, minibatch_size=maths.ceil(N), num_epochs= n, L2_lambda=L, design_matrix=design_matrix, output_data=syn_training_output_data, validation_output_data=syn_validation_output_data, validation_data=design_matrix_v, p = patience,val_size = syn_test_data.shape[0] )
		print('weight\n ', weight)
		print('\nsteps: ', steps)
		print('\nerror: ', error)
		#print('min_k_sgd',best_k_sgd)
		if min_error_sgd > error :
			
			min_error_sgd = error 
			best_k_sgd = K
			best_ita = ita
			#print('min_k_sgd',best_k_sgd,type(K))

#Get the Error of test dataset in closed form solution
print('best_k_closed_form',best_k_closed_form)
centers, spreads = find_center2(data=syn_training_data, cluster_nums=best_k_closed_form)
centers = centers[:, np.newaxis, :]
design_matrix = compute_design_matrix(X, centers, spreads)
design_matrix_v = compute_design_matrix(X_v, centers, spreads)
design_matrix_t = compute_design_matrix(X_t, centers, spreads)
closed_soln_wt = closed_form_sol(L2_lambda=L,design_matrix=design_matrix,output_data=syn_training_output_data)
print('\nTest error Closed form: ',err_rms(t=syn_test_output_data, phi=design_matrix_t, weights= closed_soln_wt, L2_lambda=L,N=syn_test_data.shape[0]) )
print('\nFinal weights',closed_soln_wt)


#Get the Error of test dataset in Gradient Descent solution
print('best_k_sgd',best_k_sgd)			
print('best_ita_sgd',best_ita)			
centers, spreads = find_center2(data=syn_training_data, cluster_nums=best_k_sgd)
centers = centers[:, np.newaxis, :]
design_matrix = compute_design_matrix(X, centers, spreads)
design_matrix_v = compute_design_matrix(X_v, centers, spreads)
design_matrix_t = compute_design_matrix(X_t, centers, spreads)
weight,error, steps = SGD_sol2(learning_rate=best_ita, minibatch_size=maths.ceil(N), num_epochs= n, L2_lambda=L, design_matrix=design_matrix, output_data=syn_training_output_data, validation_output_data=syn_validation_output_data, validation_data=design_matrix_v, p = patience,val_size = syn_test_data.shape[0] )
print('\nTest error SGD: ',err_rms(t=syn_test_output_data, phi=design_matrix_t, weights= weight, L2_lambda=L,N=syn_test_data.shape[0]) )
print('\nFinal weights',weight)
print('\nfinished SYN at',str(datetime.datetime.now()))

########################################################################################################################

#LOTOR data calculations

letor_input_data = np.genfromtxt('datafiles/Querylevelnorm_X.csv', delimiter=',')
letor_output_data = np.genfromtxt('datafiles/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])

letor_training_data = letor_input_data[:maths.ceil(0.8*letor_input_data.shape[0])]
letor_validation_data =  letor_input_data[maths.ceil(0.8*letor_input_data.shape[0]):maths.ceil(0.8*letor_input_data.shape[0]) + maths.ceil(0.1*letor_input_data.shape[0])]
letor_test_data = letor_input_data[maths.ceil(0.9*letor_input_data.shape[0]):]

letor_training_output_data = letor_output_data[:maths.ceil(0.8*letor_input_data.shape[0])]
letor_validation_output_data =  letor_output_data[maths.ceil(0.8*letor_input_data.shape[0]):maths.ceil(0.8*letor_input_data.shape[0]) + maths.ceil(0.1*letor_input_data.shape[0])]
letor_test_output_data = letor_output_data[maths.ceil(0.9*letor_input_data.shape[0]):]


N, D = letor_training_data.shape
## Assume we use 3 Gaussian basis functions M = 3
## shape = [M, 1, D]

X = letor_training_data[np.newaxis, :, :]
X_v = letor_validation_data[np.newaxis, :, :]
X_t = letor_test_data[np.newaxis, :, :]

best_k_closed_form = 1000
best_k_sgd = 1000
min_error_closed_form = 1000
min_error_sgd = 1000
best_ita = 1

for K in [3,5,10,15,20]:
	centers, spreads = find_center2(data = letor_training_data, cluster_nums = K)
	centers = centers[:, np.newaxis, :]
	design_matrix = compute_design_matrix(X, centers, spreads)
	design_matrix_v = compute_design_matrix(X_v, centers, spreads)

	closed_soln_wt = closed_form_sol(L2_lambda=L,design_matrix=design_matrix,output_data=letor_training_output_data)
	
	print('closed_form_sol',  closed_soln_wt)
	closed_form_error = err_rms(t=letor_validation_output_data, phi=design_matrix_v, weights= closed_soln_wt, L2_lambda=L, N=letor_validation_data.shape[0])
	if min_error_closed_form > closed_form_error :
		min_error_closed_form = closed_form_error
		best_k_closed_form = K
	
	for ita in [.9, .5, .1, .05, .01] :
		print(L, patience, n, ita, K )

		weight,error, steps = SGD_sol2(learning_rate=ita, minibatch_size=maths.ceil(N), num_epochs= n, L2_lambda=L, design_matrix=design_matrix, output_data=letor_training_output_data, validation_output_data=letor_validation_output_data, validation_data=design_matrix_v, p = patience,val_size = letor_test_data.shape[0] )
		print('weight\n ', weight)
		print('\nsteps: ', steps)
		print('\nerror: ', error)
		if min_error_sgd > error :
			min_error_sgd = error 
			best_k_sgd = K
			best_ita = ita


#Get the Error of test dataset in closed form solution
print('best_k_closed_form',best_k_closed_form)
centers, spreads = find_center2(data=letor_training_data, cluster_nums=best_k_closed_form)
centers = centers[:, np.newaxis, :]
design_matrix = compute_design_matrix(X, centers, spreads)
design_matrix_v = compute_design_matrix(X_v, centers, spreads)
design_matrix_t = compute_design_matrix(X_t, centers, spreads)
closed_soln_wt = closed_form_sol(L2_lambda=L,design_matrix=design_matrix,output_data=letor_training_output_data)
print('\nTest error Closed form: ',err_rms(t=letor_test_output_data, phi=design_matrix_t, weights= closed_soln_wt, L2_lambda=L,N=letor_test_data.shape[0]) )
print('\nFinal weights',closed_soln_wt)

			
centers, spreads = find_center2(data=letor_training_data, cluster_nums=best_k_sgd)
centers = centers[:, np.newaxis, :]
design_matrix = compute_design_matrix(X, centers, spreads)
design_matrix_v = compute_design_matrix(X_v, centers, spreads)
design_matrix_t = compute_design_matrix(X_t, centers, spreads)
weight,error, steps = SGD_sol2(learning_rate=best_ita, minibatch_size=maths.ceil(N), num_epochs= n, L2_lambda=L, design_matrix=design_matrix, output_data=letor_training_output_data, validation_output_data=letor_validation_output_data, validation_data=design_matrix_v, p = patience,val_size = letor_test_data.shape[0] )

print('\nTest error: ',err_rms(t=letor_test_output_data, phi=design_matrix_t, weights= weight, L2_lambda=L,N=letor_test_data.shape[0]) )
print('\nFinal weights',weight)
print('\nfinished LETOR at',str(datetime.datetime.now()))
