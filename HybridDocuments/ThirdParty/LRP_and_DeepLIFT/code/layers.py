'''
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: BSD-2-Clause
'''

import numpy as np
import pickle
from numpy import newaxis as na
from LRP_linear_layer import *

def keras_to_weights_biqrnn(orig_model, reordering):
	weights = {}

	embedding = orig_model.get_layer("embedding")		
	bidir = orig_model.get_layer("main")		
	dense = orig_model.layers[-2]
		
	weights["E"] = np.array(embedding.embeddings.container.storage[0])

	# left encoder
	SIDE_MAPPING = {"Left": "forward_layer", "Right": "backward_layer"}

	for side in SIDE_MAPPING:
		layer = getattr(bidir, SIDE_MAPPING[side])
		tmp = getattr(layer, "kernel")
		rnn_weights = np.array(tmp.container.storage[0])
				
		d = rnn_weights.shape[-1] // len(reordering)
		slices = [slice(s*d, (s+1)*d) for s in reordering]

		rnn_weights = np.concatenate([rnn_weights[:,:,s] for s in slices], axis = 2)
		
		bias = np.array(layer.bias.container.storage[0])
		bias = np.concatenate([bias[s] for s in slices], axis = 0)
		
		weights["bxh_" + side] = bias
		weights["Wxh_" + side] = rnn_weights

	dense_weights = np.array(dense.kernel.container.storage[0])
	dense_bias = np.array(dense.bias.container.storage[0])
	split = dense_weights.shape[0] // 2

	weights["Why_Left"]  = dense_weights[:split].transpose()
	weights["Why_Right"] = dense_weights[split:].transpose()
	weights["bhy_Left"] = dense_bias / 2
	weights["bhy_Right"] = dense_bias / 2
	
	return weights

def keras_to_weights_cnn(orig_model):
	embedding = orig_model.get_layer("embedding")		
	dense = orig_model.get_layer("dense_1")
	
	weights = {}	
	weights["E"] = orig_model.get_weights()[0]
	weights["Why"] = orig_model.get_weights()[-2].transpose()
	weights["bhy"] = orig_model.get_weights()[-1]
	
	weights["W"] = orig_model.get_weights()[1]
	weights["B"] = orig_model.get_weights()[2]

	return weights

def keras_to_weights_birnn(orig_model, reordering):
	weights = {}

	embedding = orig_model.get_layer("embedding")		
	bidir = orig_model.get_layer("main")		
	dense = orig_model.get_layer("dense_1")
		
	weights["E"] = np.array(embedding.embeddings.container.storage[0])

	# left encoder
	SIDE_MAPPING = {"Left": "forward_layer", "Right": "backward_layer"}
	
	CONNECTION_MAPPING = {"x": "kernel", "h": "recurrent_kernel"}
	

	for side in SIDE_MAPPING:
		for connection in CONNECTION_MAPPING:
			layer = getattr(bidir, SIDE_MAPPING[side])
			tmp = getattr(layer, CONNECTION_MAPPING[connection])
			rnn_weights = np.array(tmp.container.storage[0])
				
			d = rnn_weights.shape[1] // len(reordering)
			slices = [slice(s*d, (s+1)*d) for s in reordering]

			rnn_weights = np.concatenate([rnn_weights[:,s] for s in slices], axis = 1)
			if connection == "h":
				rnn_weights = np.concatenate([rnn_weights[s] for s in slices], axis = 0)
					
			weights["W" + connection + "h_" + side] = np.transpose(rnn_weights)

		bias = np.array(layer.bias.container.storage[0])
		bias = np.concatenate([bias[s] for s in slices], axis = 0)
		
		weights["bxh_" + side] = bias / 2
		weights["bhh_" + side] = bias / 2

	dense_weights = np.array(dense.kernel.container.storage[0])
	dense_bias = np.array(dense.bias.container.storage[0])
	split = dense_weights.shape[0] // 2

	weights["Why_Left"]  = dense_weights[:split].transpose()
	weights["Why_Right"] = dense_weights[split:].transpose()
	weights["bhy_Left"] = dense_bias / 2
	weights["bhy_Right"] = dense_bias / 2
	
	return weights


class CNN_np:
	def __init__(self, weights, mode):
		for key in weights:
			setattr(self, key, weights[key])
	
		assert mode in ("dl", "lrp", "dlrec")
		
		self.mode = mode
		if mode == "dl":
			self.ref = CNN_np(weights, "dlrec")
	
	def set_input(self, w, delete_pos=None):
		"""
		Build the numerical input x/x_rev from word sequence indices w (+ initialize hidden layers h, c)
		Optionally delete words at positions delete_pos.
		"""
		T	  = w.shape[-1]					# input word sequence length
		d	  = int(self.W.shape[-1])  # hidden layer dimension
		f = int(self.W.shape[0])
		e	  = self.E.shape[1]				# word embedding dimension
		x = np.zeros((T, e))
		x[:,:] = self.E[w,:]
		if delete_pos is not None:
			x[delete_pos, :] = np.zeros((len(delete_pos), e))
		
		self.w			  = w
		self.x			  = np.concatenate([np.zeros(((f-1)//2, e)), x, np.zeros(((f-1)//2, e))], axis = 0)
		self.g		 = np.zeros((T, d))
		self.g_pre		 = np.zeros((T, d))
	 
   
	def forward(self):
		"""
		Update the hidden layer values (using model weights and numerical input x/x_rev previously built from word sequence w)
		"""
		T	  = self.w.shape[0]
		d	  = int(self.W.shape[-1])	  
		f 	= int(self.W.shape[0])
		
		for t in range(T): 
			for c in range(f):
				self.g_pre[t] += np.dot(self.x[t+f-1-c], self.W[c])
			self.g_pre[t] += self.B
			self.g[t] = np.maximum(self.g_pre[t], np.zeros(d))
		
		self.h = self.g.max(axis = 0)

		self.s  = np.dot(self.Why,  self.h) + self.bhy
		self.pred = np.exp(self.s) / np.sum(np.exp(self.s))
		return self.s.copy() # prediction scores
	 
	def predict(self, w):
		self.set_input(w)
		self.forward()
		return self.pred
		
	def lrp(self, w, LRP_class, eps=0.001, bias_factor=1.0):
		"""
		Update the hidden layer relevances by performing LRP for the target class LRP_class
		"""
		# forward pass
		self.set_input(w)
		self.forward() 

		if self.mode == "dl":
			self.ref.set_input(w)
			self.ref.x = np.zeros_like(self.x)
			self.ref.x_rev = np.zeros_like(self.x)
			
			self.ref.forward()
			
		T	  = self.w.shape[0]
		d	  = int(self.W.shape[-1])
		e	  = self.E.shape[1] 
		C	  = self.Why.shape[0]  # number of classes
		f = int(self.W.shape[0])

		# initialize
		Rx	   = np.zeros(self.x.shape)
		
		Rh  = np.zeros((d,))
		Rg  = np.zeros((T, d)) # gate g only
		
		Rout_mask			= np.zeros((C))
		Rout_mask[LRP_class] = 1.0  
		
		# format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)

		s = self.s.copy()
		h = self.h.copy()
		g = self.g.copy()
		g_pre = self.g_pre.copy()
		
		if self.mode == "dl":
			s -= self.ref.s
			h -= self.ref.h
			g -= self.ref.g
			g_pre -= self.ref.g_pre

		Rh  = lrp_linear(h,  self.Why.T , np.zeros((C)), s, s*Rout_mask, d, eps, bias_factor, debug=False)
		for feature in range(d):
			Rg[g[:,feature].argmax()][feature] = Rh[feature]

		for t in range(T):
			for c in range(f):
				Rx[t+f-1-c] += lrp_linear(self.x[t+f-1-c], self.W[c], self.B, g_pre[t], Rg[t], f*e, eps, bias_factor, debug=False)
		a, b, c = Rx[(f-1)//2:-(f-1)//2], 0, Rx[:(f-1)//2].sum() + Rx[-(f-1)//2:].sum()

		return a, b, c


class BIQGRU_np:
	def __init__(self, weights, mode):
		for key in weights:
			setattr(self, key, weights[key])
	
		assert mode in ("dl", "lrp", "dlrec")
		
		self.mode = mode
		if mode == "dl":
			self.ref = BIQGRU_np(weights, "dlrec")
	
	def set_input(self, w, delete_pos=None):
		"""
		Build the numerical input x/x_rev from word sequence indices w (+ initialize hidden layers h, c)
		Optionally delete words at positions delete_pos.
		"""
		T	  = w.shape[-1]					# input word sequence length
		d	  = int(self.Wxh_Left.shape[-1]/2)  # hidden layer dimension
		f = int(self.Wxh_Left.shape[0])
		e	  = self.E.shape[1]				# word embedding dimension
		x = np.zeros((T, e))
		x[:,:] = self.E[w,:]
		if delete_pos is not None:
			x[delete_pos, :] = np.zeros((len(delete_pos), e))
		
		self.w			  = w
		self.x			  = np.concatenate([np.zeros((f-1, e)), x], axis = 0)
		self.x_rev		  = np.concatenate([np.zeros((f-1, e)), x[::-1].copy()], axis = 0)
	
		self.h_Left		 = np.zeros((T, d))
		self.h_Right		= np.zeros((T, d))
	 
   
	def forward(self):
		"""
		Update the hidden layer values (using model weights and numerical input x/x_rev previously built from word sequence w)
		"""
		T	  = self.w.shape[0]
		d	  = int(self.Wxh_Left.shape[-1]/2)	  
		f = int(self.Wxh_Left.shape[0])
		
		# initialize
		self.gates_pre_Left = np.zeros((T, 2*d))  # gates i, g, f, o pre-activation
		self.gates_Left	 = np.zeros((T, 2*d))  # gates i, g, f, o activation
		
		self.gates_pre_Right= np.zeros((T, 2*d))
		self.gates_Right	= np.zeros((T, 2*d)) 
		
		for t in range(T): 
			for c in range(f):
				self.gates_pre_Left[t] += np.dot(self.x[t+f-1-c], self.Wxh_Left[c])
			self.gates_pre_Left[t] += + self.bxh_Left
			zeros = np.zeros_like(self.gates_pre_Left[t,:d])
			ones = zeros + 1
			self.gates_Left[t,:d] = np.maximum(zeros, np.minimum(ones, 0.2*self.gates_pre_Left[t,:d] + 0.5))
			self.gates_Left[t,d:2*d] = np.tanh(self.gates_pre_Left[t,d:2*d]) 
			self.h_Left[t] = self.gates_Left[t,:d]*self.h_Left[t-1] + (1-self.gates_Left[t,:d]) * self.gates_Left[t,d:]
			
			for c in range(f):
				self.gates_pre_Right[t] += np.dot(self.x_rev[t+f-1-c], self.Wxh_Right[c])
			self.gates_pre_Right[t] += + self.bxh_Right
			zeros = np.zeros_like(self.gates_pre_Right[t,:d])
			ones = zeros + 1
			self.gates_Right[t,:d] = np.maximum(zeros, np.minimum(ones, 0.2*self.gates_pre_Right[t,:d] + 0.5))
			self.gates_Right[t,d:2*d] = np.tanh(self.gates_pre_Right[t,d:2*d]) 
			self.h_Right[t] = self.gates_Right[t,:d]*self.h_Right[t-1] + (1-self.gates_Right[t,:d]) * self.gates_Right[t,d:]
			
		self.y_Left  = np.dot(self.Why_Left,  self.h_Left[T-1]) + self.bhy_Left
		self.y_Right = np.dot(self.Why_Right, self.h_Right[T-1]) + self.bhy_Right
		self.s	   = self.y_Left + self.y_Right
		
		self.pred = np.exp(self.s) / np.sum(np.exp(self.s))
		return self.s.copy() # prediction scores
	 
	def predict(self, w):
		self.set_input(w)
		self.forward()
		return self.pred
		
	def lrp(self, w, LRP_class, eps=0.001, bias_factor=1.0):
		"""
		Update the hidden layer relevances by performing LRP for the target class LRP_class
		"""
		# forward pass
		self.set_input(w)
		self.forward() 

		if self.mode == "dl":
			self.ref.set_input(w)
			self.ref.x = np.zeros_like(self.x)
			self.ref.x_rev = np.zeros_like(self.x)
			
			self.ref.forward()
			
		T	  = self.w.shape[0]
		d	  = int(self.Wxh_Left.shape[-1]/2)
		e	  = self.E.shape[1] 
		C	  = self.Why_Left.shape[0]  # number of classes
		f = int(self.Wxh_Left.shape[0])

		# initialize
		Rx	   = np.zeros(self.x.shape)
		Rx_rev   = np.zeros(self.x.shape)
		
		Rh_Left  = np.zeros((T+1, d))
		Rg_Left  = np.zeros((T,   d)) # gate g only
		Rh_Right = np.zeros((T+1, d))
		Rg_Right = np.zeros((T,   d)) # gate g only
		
		Rout_mask			= np.zeros((C))
		Rout_mask[LRP_class] = 1.0  
		
		# format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)

		s = self.s.copy()
		
		gates_Left = self.gates_Left.copy()
		gates_pre_Left = self.gates_pre_Left.copy()
		h_Left = self.h_Left.copy()
		
		gates_Right = self.gates_Right.copy()
		gates_pre_Right = self.gates_pre_Right.copy()
		h_Right = self.h_Right.copy()
		
		if self.mode == "dl":
			s -= self.ref.s

			gates_Left[:,d:] -= self.ref.gates_Left[:,d:]
			gates_pre_Left[:,d:] -= self.ref.gates_pre_Left[:,d:]
			h_Left -= self.ref.h_Left
		
			gates_Right[:,d:] -= self.ref.gates_Right[:,d:]
			gates_pre_Right[:,d:] -= self.ref.gates_pre_Right[:,d:]
			h_Right -= self.ref.h_Right

			
		Rh_Left[T-1]  = lrp_linear(h_Left[T-1],  self.Why_Left.T , np.zeros((C)), s, s*Rout_mask, 2*d, eps, bias_factor, debug=False)
		Rh_Right[T-1] = lrp_linear(h_Right[T-1], self.Why_Right.T, np.zeros((C)), s, s*Rout_mask, 2*d, eps, bias_factor, debug=False)
		for t in reversed(range(T)):
			Rh_Left[t-1]  = lrp_linear(gates_Left[t,:d]*h_Left[t-1], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor, debug=False)
			Rg_Left[t]	= lrp_linear((1-gates_Left[t,:d])*gates_Left[t,d:], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor, debug=False)
			
			Rh_Right[t-1]  = lrp_linear(gates_Right[t,:d]*h_Right[t-1], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor, debug=False)
			Rg_Right[t]	= lrp_linear((1-gates_Right[t,:d])*gates_Right[t,d:], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor, debug=False)
			
		for t in range(T):
			for c in range(f):
				Rx[t+f-1-c] += lrp_linear(self.x[t+f-1-c], self.Wxh_Left[c,:,d:], self.bxh_Left[d:], gates_pre_Left[t,d:], Rg_Left[t], f*e, eps, bias_factor, debug=False)
				Rx_rev[t+f-1-c] += lrp_linear(self.x_rev[t+f-1-c], self.Wxh_Right[c, :, d:], self.bxh_Right[d:], gates_pre_Right[t,d:], Rg_Right[t], f*e, eps, bias_factor, debug=False)
			
		a, b, c = Rx[f-1:], Rx_rev[f-1:][::-1,:], Rh_Left[-1].sum()+Rh_Right[-1].sum()+ + Rx[:f-1].sum() + Rx_rev[:f-1][::-1].sum()

		summed = np.sum(a+b, axis = -1)
		relpeak = summed.argmax(-1) / len(summed)
		return a, b, c

class BIQLSTM_np:
	def __init__(self, weights, mode):
		for key in weights:
			setattr(self, key, weights[key])

		assert mode in ("dl", "lrp", "dlrec")
		
		self.mode = mode
		if mode == "dl":
			self.ref = BIQLSTM_np(weights, "dlrec")
	
	def set_input(self, w, delete_pos=None):
		"""
		Build the numerical input x/x_rev from word sequence indices w (+ initialize hidden layers h, c)
		Optionally delete words at positions delete_pos.
		"""
		T	  = w.shape[-1]					# input word sequence length
		d	  = int(self.Wxh_Left.shape[-1]/4)  # hidden layer dimension
		f = int(self.Wxh_Left.shape[0])
		e	  = self.E.shape[1]				# word embedding dimension
		x = np.zeros((T, e))
		x[:,:] = self.E[w,:]
		if delete_pos is not None:
			x[delete_pos, :] = np.zeros((len(delete_pos), e))
		
		self.w			  = w
		self.x			  = np.concatenate([np.zeros((f-1, e)), x], axis = 0)
		self.x_rev		  = np.concatenate([np.zeros((f-1, e)), x[::-1].copy()], axis = 0)
	
		self.h_Left		 = np.zeros((T, d))
		self.c_Left		 = np.zeros((T, d))
		self.h_Right		= np.zeros((T, d))
		self.c_Right		= np.zeros((T, d))
	
   
	def forward(self):
		"""
		Update the hidden layer values (using model weights and numerical input x/x_rev previously built from word sequence w)
		"""
		T	  = self.w.shape[0]
		d	  = int(self.Wxh_Left.shape[-1]/4)	  
		idx	= np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of the gates i,f,o
		f = int(self.Wxh_Left.shape[0])
		
		# initialize
		self.gates_pre_Left = np.zeros((T, 4*d))  # gates i, g, f, o pre-activation
		self.gates_Left	 = np.zeros((T, 4*d))  # gates i, g, f, o activation
		
		self.gates_pre_Right= np.zeros((T, 4*d))
		self.gates_Right	= np.zeros((T, 4*d)) 
			 
		for t in range(T): 
			for c in range(f):
				self.gates_pre_Left[t] += np.dot(self.x[t+f-1-c], self.Wxh_Left[c])
			self.gates_pre_Left[t] += + self.bxh_Left
			zeros = np.zeros_like(self.gates_pre_Left[t,idx])
			ones = zeros + 1
			self.gates_Left[t,idx] = np.maximum(zeros, np.minimum(ones, 0.2*self.gates_pre_Left[t,idx] + 0.5))
			self.gates_Left[t,d:2*d] = np.tanh(self.gates_pre_Left[t,d:2*d]) 
			self.c_Left[t] = self.gates_Left[t,2*d:3*d]*self.c_Left[t-1] + self.gates_Left[t,0:d]*self.gates_Left[t,d:2*d]
			self.h_Left[t] = self.gates_Left[t,3*d:4*d]*np.tanh(self.c_Left[t])
			
			for c in range(f):
				self.gates_pre_Right[t] += np.dot(self.x_rev[t+f-1-c], self.Wxh_Right[c])
			self.gates_pre_Right[t] += + self.bxh_Right
			zeros = np.zeros_like(self.gates_pre_Right[t,idx])
			ones = zeros + 1
			self.gates_Right[t,idx] = np.maximum(zeros, np.minimum(ones, 0.2*self.gates_pre_Right[t,idx] + 0.5))
			self.gates_Right[t,d:2*d] = np.tanh(self.gates_pre_Right[t,d:2*d]) 
			self.c_Right[t] = self.gates_Right[t,2*d:3*d]*self.c_Right[t-1] + self.gates_Right[t,0:d]*self.gates_Right[t,d:2*d]
			self.h_Right[t] = self.gates_Right[t,3*d:4*d]*np.tanh(self.c_Right[t])
			
		self.y_Left  = np.dot(self.Why_Left,  self.h_Left[T-1]) + self.bhy_Left
		self.y_Right = np.dot(self.Why_Right, self.h_Right[T-1]) + self.bhy_Right
		self.s	   = self.y_Left + self.y_Right
		
		self.pred = np.exp(self.s) / np.sum(np.exp(self.s))
		return self.s.copy() # prediction scores
	 
	def predict(self, w):
		self.set_input(w)
		self.forward()
		return self.pred
		
	def lrp(self, w, LRP_class, eps=0.001, bias_factor=1.0):
		"""
		Update the hidden layer relevances by performing LRP for the target class LRP_class
		"""
		# forward pass
		self.set_input(w)
		self.forward() 

		if self.mode == "dl":
			self.ref.set_input(w)
			self.ref.x = np.zeros_like(self.x)
			self.ref.x_rev = np.zeros_like(self.x)
			
			self.ref.forward()
			
		T	  = self.w.shape[0]
		d	  = int(self.Wxh_Left.shape[-1]/4)
		e	  = self.E.shape[1] 
		C	  = self.Why_Left.shape[0]  # number of classes
		idx	= np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) 
		f = int(self.Wxh_Left.shape[0])

		# initialize
		Rx	   = np.zeros(self.x.shape)
		Rx_rev   = np.zeros(self.x.shape)
		
		Rh_Left  = np.zeros((T+1, d))
		Rc_Left  = np.zeros((T+1, d))
		Rg_Left  = np.zeros((T,   d)) # gate g only
		Rh_Right = np.zeros((T+1, d))
		Rc_Right = np.zeros((T+1, d))
		Rg_Right = np.zeros((T,   d)) # gate g only
		
		Rout_mask			= np.zeros((C))
		Rout_mask[LRP_class] = 1.0  
		
		# format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)

		s = self.s.copy()
		
		gates_Left = self.gates_Left.copy()
		gates_pre_Left = self.gates_pre_Left.copy()
		c_Left = self.c_Left.copy()
		h_Left = self.h_Left.copy()
		
		gates_Right = self.gates_Right.copy()
		gates_pre_Right = self.gates_pre_Right.copy()
		c_Right = self.c_Right.copy()
		h_Right = self.h_Right.copy()
		
		if self.mode == "dl":
			s -= self.ref.s

			gates_Left[:,d:2*d] -= self.ref.gates_Left[:,d:2*d]
			gates_pre_Left[:,d:2*d] -= self.ref.gates_pre_Left[:,d:2*d]
			c_Left -= self.ref.c_Left
			h_Left -= self.ref.h_Left
		
			gates_Right[:,d:2*d] -= self.ref.gates_Right[:,d:2*d]
			gates_pre_Right[:,d:2*d] -= self.ref.gates_pre_Right[:,d:2*d]
			c_Right -= self.ref.c_Right
			h_Right -= self.ref.h_Right

			
		Rh_Left[T-1]  = lrp_linear(h_Left[T-1],  self.Why_Left.T , np.zeros((C)), s, s*Rout_mask, 2*d, eps, bias_factor, debug=False)
		Rh_Right[T-1] = lrp_linear(h_Right[T-1], self.Why_Right.T, np.zeros((C)), s, s*Rout_mask, 2*d, eps, bias_factor, debug=False)
		for t in reversed(range(T)):
			#Rc_Left[t]   += Rh_Left[t]
			Rc_Left[t]   += lrp_linear(gates_Left[t,3*d:]*np.tanh(c_Left[t]), np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], d, eps, bias_factor, debug=False)

			Rc_Left[t-1]  = lrp_linear(gates_Left[t,2*d:3*d]*c_Left[t-1], np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
			Rg_Left[t]	= lrp_linear(gates_Left[t,0:d]*gates_Left[t,d:2*d], np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
			
			#Rc_Right[t]   += Rh_Right[t]
			Rc_Right[t]   += lrp_linear(gates_Right[t,3*d:]*np.tanh(c_Right[t]), np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], d, eps, bias_factor, debug=False)
			Rc_Right[t-1]  = lrp_linear(gates_Right[t,2*d:3*d]*c_Right[t-1], np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor, debug=False)
			Rg_Right[t]	= lrp_linear(gates_Right[t,0:d]*gates_Right[t,d:2*d], np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor, debug=False)
			
		for t in range(T):
			for c in range(f):
				Rx[t+f-1-c] += lrp_linear(self.x[t+f-1-c], self.Wxh_Left[c,:,d:2*d], self.bxh_Left[d:2*d], gates_pre_Left[t,d:2*d], Rg_Left[t], f*e, eps, bias_factor, debug=False)
				Rx_rev[t+f-1-c] += lrp_linear(self.x_rev[t+f-1-c], self.Wxh_Right[c,:,d:2*d], self.bxh_Right[d:2*d], gates_pre_Right[t,d:2*d], Rg_Right[t], f*e, eps, bias_factor, debug=False)
			
		a, b, c = Rx[f-1:], Rx_rev[f-1:][::-1,:], Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum() + Rx[:f-1].sum() + Rx_rev[:f-1][::-1].sum()

		summed = np.sum(a+b, axis = -1)
		relpeak = summed.argmax(-1) / len(summed)
		return a, b, c


class BILSTM_np:
	def __init__(self, weights, mode):
		for key in weights:
			setattr(self, key, weights[key])

		assert mode in ("dl", "lrp", "dlrec")
		
		self.mode = mode
		if mode == "dl":
			self.ref = BILSTM_np(weights, "dlrec")
	
	def set_input(self, w, delete_pos=None):
		"""
		Build the numerical input x/x_rev from word sequence indices w (+ initialize hidden layers h, c)
		Optionally delete words at positions delete_pos.
		"""
		T	  = w.shape[-1]					# input word sequence length
		d	  = int(self.Wxh_Left.shape[0]/4)  # hidden layer dimension
		e	  = self.E.shape[1]				# word embedding dimension
		x = np.zeros((T, e))
		x[:,:] = self.E[w,:]
		if delete_pos is not None:
			x[delete_pos, :] = np.zeros((len(delete_pos), e))
		
		self.w			  = w
		self.x			  = x
		self.x_rev		  = x[::-1,:].copy()
		
		self.h_Left		 = np.zeros((T+1, d))
		self.c_Left		 = np.zeros((T+1, d))
		self.h_Right		= np.zeros((T+1, d))
		self.c_Right		= np.zeros((T+1, d))
	 
   
	def forward(self):
		"""
		Update the hidden layer values (using model weights and numerical input x/x_rev previously built from word sequence w)
		"""
		T	  = self.x.shape[0]
		d	  = int(self.Wxh_Left.shape[0]/4)	  
		idx	= np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of the gates i,f,o
		  
		# initialize
		self.gates_xh_Left  = np.zeros((T, 4*d))  
		self.gates_hh_Left  = np.zeros((T, 4*d)) 
		self.gates_pre_Left = np.zeros((T, 4*d))  # gates i, g, f, o pre-activation
		self.gates_Left	 = np.zeros((T, 4*d))  # gates i, g, f, o activation
		
		self.gates_xh_Right = np.zeros((T, 4*d))  
		self.gates_hh_Right = np.zeros((T, 4*d)) 
		self.gates_pre_Right= np.zeros((T, 4*d))
		self.gates_Right	= np.zeros((T, 4*d)) 
			 
		for t in range(T): 
			self.gates_xh_Left[t]	= np.dot(self.Wxh_Left, self.x[t]) + self.bxh_Left
			self.gates_hh_Left[t]	= np.dot(self.Whh_Left, self.h_Left[t-1]) + self.bhh_Left
			self.gates_pre_Left[t]   = self.gates_xh_Left[t] + self.gates_hh_Left[t]
			#self.gates_Left[t,idx]   = 1.0/(1.0 + np.exp(- self.gates_pre_Left[t,idx]))
			zeros = np.zeros_like(self.gates_pre_Left[t,idx])
			ones = zeros + 1
			self.gates_Left[t,idx] = np.maximum(zeros, np.minimum(ones, 0.2*self.gates_pre_Left[t,idx] + 0.5))
			self.gates_Left[t,d:2*d] = np.tanh(self.gates_pre_Left[t,d:2*d]) 
			self.c_Left[t]		   = self.gates_Left[t,2*d:3*d]*self.c_Left[t-1] + self.gates_Left[t,0:d]*self.gates_Left[t,d:2*d]
			self.h_Left[t]		   = self.gates_Left[t,3*d:4*d]*np.tanh(self.c_Left[t])
			
			self.gates_xh_Right[t]	= np.dot(self.Wxh_Right, self.x_rev[t]) + self.bxh_Right
			self.gates_hh_Right[t]	= np.dot(self.Whh_Right, self.h_Right[t-1]) + self.bhh_Right
			self.gates_pre_Right[t]   = self.gates_xh_Right[t] + self.gates_hh_Right[t]
			#self.gates_Right[t,idx]   = 1.0/(1.0 + np.exp(- self.gates_pre_Right[t,idx]))
			zeros = np.zeros_like(self.gates_pre_Right[t,idx])
			ones = zeros + 1
			self.gates_Right[t,idx] = np.maximum(zeros, np.minimum(ones, 0.2*self.gates_pre_Right[t,idx] + 0.5))
			self.gates_Right[t,d:2*d] = np.tanh(self.gates_pre_Right[t,d:2*d])				 
			self.c_Right[t]		   = self.gates_Right[t,2*d:3*d]*self.c_Right[t-1] + self.gates_Right[t,0:d]*self.gates_Right[t,d:2*d]
			self.h_Right[t]		   = self.gates_Right[t,3*d:4*d]*np.tanh(self.c_Right[t])
			
		self.y_Left  = np.dot(self.Why_Left,  self.h_Left[T-1]) + self.bhy_Left
		self.y_Right = np.dot(self.Why_Right, self.h_Right[T-1]) + self.bhy_Right
		self.s	   = self.y_Left + self.y_Right
		
		self.pred = np.exp(self.s) / np.sum(np.exp(self.s))
		return self.s.copy() # prediction scores
	 
	def predict(self, w):
		self.set_input(w)
		self.forward()
		return self.pred
		
	def lrp(self, w, LRP_class, eps=0.001, bias_factor=1.0):
		"""
		Update the hidden layer relevances by performing LRP for the target class LRP_class
		"""
		# forward pass
		self.set_input(w)
		self.forward() 

		if self.mode == "dl":
			self.ref.set_input(w)
			self.ref.x = np.zeros_like(self.x)
			self.ref.x_rev = np.zeros_like(self.x)
			
			self.ref.forward()
			
		T	  = self.w.shape[-1]
		d	  = int(self.Wxh_Left.shape[0]/4)
		e	  = self.E.shape[1] 
		C	  = self.Why_Left.shape[0]  # number of classes
		idx	= np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) 
		
		# initialize
		Rx	   = np.zeros(self.x.shape)
		Rx_rev   = np.zeros(self.x.shape)
		
		Rh_Left  = np.zeros((T+1, d))
		Rc_Left  = np.zeros((T+1, d))
		Rg_Left  = np.zeros((T,   d)) # gate g only
		Rh_Right = np.zeros((T+1, d))
		Rc_Right = np.zeros((T+1, d))
		Rg_Right = np.zeros((T,   d)) # gate g only
		
		Rout_mask			= np.zeros((C))
		Rout_mask[LRP_class] = 1.0  
		
		# format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)

		s = self.s.copy()
		
		gates_Left = self.gates_Left.copy()
		gates_pre_Left = self.gates_pre_Left.copy()
		c_Left = self.c_Left.copy()
		h_Left = self.h_Left.copy()
		
		gates_Right = self.gates_Right.copy()
		gates_pre_Right = self.gates_pre_Right.copy()
		c_Right = self.c_Right.copy()
		h_Right = self.h_Right.copy()
		
		if self.mode == "dl":
			s -= self.ref.s

			gates_Left[:,d:2*d] -= self.ref.gates_Left[:,d:2*d]
			gates_pre_Left[:,d:2*d] -= self.ref.gates_pre_Left[:,d:2*d]
			c_Left -= self.ref.c_Left
			h_Left -= self.ref.h_Left
		
			gates_Right[:,d:2*d] -= self.ref.gates_Right[:,d:2*d]
			gates_pre_Right[:,d:2*d] -= self.ref.gates_pre_Right[:,d:2*d]
			c_Right -= self.ref.c_Right
			h_Right -= self.ref.h_Right

			
		Rh_Left[T-1]  = lrp_linear(h_Left[T-1],  self.Why_Left.T , np.zeros((C)), s, s*Rout_mask, 2*d, eps, bias_factor, debug=False)
		Rh_Right[T-1] = lrp_linear(h_Right[T-1], self.Why_Right.T, np.zeros((C)), s, s*Rout_mask, 2*d, eps, bias_factor, debug=False)
		for t in reversed(range(T)):
			#Rc_Left[t]   += Rh_Left[t]
			Rc_Left[t]   += lrp_linear(gates_Left[t,3*d:]*np.tanh(c_Left[t]), np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], d, eps, bias_factor, debug=False)

			Rc_Left[t-1]  = lrp_linear(gates_Left[t,2*d:3*d]*c_Left[t-1], np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
			Rg_Left[t]	= lrp_linear(gates_Left[t,0:d]*gates_Left[t,d:2*d], np.identity(d), np.zeros((d)), c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
			Rx[t]		 = lrp_linear(self.x[t], self.Wxh_Left[d:2*d].T, self.bxh_Left[d:2*d]+self.bhh_Left[d:2*d], gates_pre_Left[t,d:2*d], Rg_Left[t], d+e, eps, bias_factor, debug=False)
			Rh_Left[t-1]  = lrp_linear(h_Left[t-1], self.Whh_Left[d:2*d].T, self.bxh_Left[d:2*d]+self.bhh_Left[d:2*d], gates_pre_Left[t,d:2*d], Rg_Left[t], d+e, eps, bias_factor, debug=False)
			
			#Rc_Right[t]  += Rh_Right[t]
			Rc_Right[t]   += lrp_linear(gates_Right[t,3*d:]*np.tanh(c_Right[t]), np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], d, eps, bias_factor, debug=False)
			Rc_Right[t-1] = lrp_linear(gates_Right[t,2*d:3*d]*c_Right[t-1],	 np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor, debug=False)
			Rg_Right[t]   = lrp_linear(gates_Right[t,0:d]*gates_Right[t,d:2*d], np.identity(d), np.zeros((d)), c_Right[t], Rc_Right[t], 2*d, eps, bias_factor, debug=False)
			Rx_rev[t]	 = lrp_linear(self.x_rev[t],	 self.Wxh_Right[d:2*d].T, self.bxh_Right[d:2*d]+self.bhh_Right[d:2*d], gates_pre_Right[t,d:2*d], Rg_Right[t], d+e, eps, bias_factor, debug=False)
			Rh_Right[t-1] = lrp_linear(h_Right[t-1], self.Whh_Right[d:2*d].T, self.bxh_Right[d:2*d]+self.bhh_Right[d:2*d], gates_pre_Right[t,d:2*d], Rg_Right[t], d+e, eps, bias_factor, debug=False)
	
		a, b, c = Rx, Rx_rev[::-1,:], Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum()

		return a, b, c


class BIGRU_np:
	def __init__(self, weights, mode):
		for key in weights:
			setattr(self, key, weights[key])

		assert mode in ("dl", "lrp", "dlrec")
		
		self.mode = mode
		if mode == "dl":
			self.ref = BIGRU_np(weights, "dlrec")
	
	def set_input(self, w, delete_pos=None):
		"""
		Build the numerical input x/x_rev from word sequence indices w (+ initialize hidden layers h, c)
		Optionally delete words at positions delete_pos.
		"""
		T	  = w.shape[-1]					# input word sequence length
		d	  = int(self.Wxh_Left.shape[0]/3)  # hidden layer dimension
		e	  = self.E.shape[1]				# word embedding dimension
		x = np.zeros((T, e))
		x[:,:] = self.E[w,:]
		if delete_pos is not None:
			x[delete_pos, :] = np.zeros((len(delete_pos), e))
		
		self.w			  = w
		self.x			  = x
		self.x_rev		  = x[::-1,:].copy()
		
		self.h_Left		 = np.zeros((T+1, d))
		self.c_Left		 = np.zeros((T+1, d))
		self.h_Right		= np.zeros((T+1, d))
		self.c_Right		= np.zeros((T+1, d))
	 
   
	def forward(self):
		"""
		Update the hidden layer values (using model weights and numerical input x/x_rev previously built from word sequence w)
		"""
		T	  = self.x.shape[0]
		d	  = int(self.Wxh_Left.shape[0]/3)	  
		idx	= np.hstack((np.arange(0,d), np.arange(d,2*d))).astype(int) # indices of the gates i,f,o
		  
		# initialize
		self.gates_xh_Left  = np.zeros((T, 3*d))  
		self.gates_pre_Left = np.zeros((T, 3*d))  # gates i, g, f, o pre-activation
		self.gates_Left	 = np.zeros((T, 3*d))  # gates i, g, f, o activation
		
		self.gates_xh_Right = np.zeros((T, 3*d))  
		self.gates_pre_Right= np.zeros((T, 3*d))
		self.gates_Right	= np.zeros((T, 3*d)) 
			 
		for t in range(T):
			
			def hard_sigmoid(vec):
				zeros = np.zeros_like(vec)
				ones = zeros + 1
				return np.maximum(zeros, np.minimum(ones, 0.2 * vec + 0.5))
 
			self.gates_xh_Left[t]	= np.dot(self.Wxh_Left, self.x[t]) + self.bxh_Left
			
			z_Left = self.gates_xh_Left[t][:d] + np.dot(self.Whh_Left[:d], self.h_Left[t-1]) + self.bhh_Left[:d]
			r_Left = self.gates_xh_Left[t][d:2*d] + np.dot(self.Whh_Left[d:2*d], self.h_Left[t-1]) + self.bhh_Left[d:2*d]

			hh_Left = self.gates_xh_Left[t][2*d:] + np.dot(self.Whh_Left[2*d:], hard_sigmoid(r_Left) * self.h_Left[t-1]) + self.bhh_Left[2*d:]
			
			self.gates_pre_Left[t] = np.concatenate([z_Left, r_Left, hh_Left], axis = -1)
			self.gates_Left[t, idx] = hard_sigmoid(self.gates_pre_Left[t, idx])
			self.gates_Left[t, 2*d:] = np.tanh(self.gates_pre_Left[t, 2*d:])
			self.h_Left[t]		   = self.gates_Left[t,:d]*self.h_Left[t-1]+(1-self.gates_Left[t,:d])*self.gates_Left[t,2*d:]
	
		
			self.gates_xh_Right[t]	= np.dot(self.Wxh_Right, self.x_rev[t]) + self.bxh_Right
			
			z_Right = self.gates_xh_Right[t][:d] + np.dot(self.Whh_Right[:d], self.h_Right[t-1]) + self.bhh_Right[:d]
			r_Right = self.gates_xh_Right[t][d:2*d] + np.dot(self.Whh_Right[d:2*d], self.h_Right[t-1]) + self.bhh_Right[d:2*d]

			hh_Right = self.gates_xh_Right[t][2*d:] + np.dot(self.Whh_Right[2*d:], hard_sigmoid(r_Right) * self.h_Right[t-1]) + self.bhh_Right[2*d:]
			
			self.gates_pre_Right[t] = np.concatenate([z_Right, r_Right, hh_Right], axis = -1)
			self.gates_Right[t, idx] = hard_sigmoid(self.gates_pre_Right[t, idx])
			self.gates_Right[t, 2*d:] = np.tanh(self.gates_pre_Right[t, 2*d:])
			self.h_Right[t]		   = self.gates_Right[t,:d]*self.h_Right[t-1]+(1-self.gates_Right[t,:d])*self.gates_Right[t,2*d:]
			
		self.y_Left  = np.dot(self.Why_Left,  self.h_Left[T-1]) + self.bhy_Left
		self.y_Right = np.dot(self.Why_Right, self.h_Right[T-1]) + self.bhy_Right
		self.s	   = self.y_Left + self.y_Right
		
		self.pred = np.exp(self.s) / np.sum(np.exp(self.s))
		return self.s.copy() # prediction scores
	 
	def predict(self, w):
		self.set_input(w)
		self.forward()
		return self.pred
		
	def lrp(self, w, LRP_class, eps=0.001, bias_factor=1.0):
		"""
		Update the hidden layer relevances by performing LRP for the target class LRP_class
		"""
		# forward pass
		self.set_input(w)
		self.forward() 
		
		if self.mode == "dl":
			self.ref.set_input(w)
			self.ref.x = np.zeros_like(self.x)
			self.ref.x_rev = np.zeros_like(self.x)
		
			self.ref.forward()
		
		T	  = self.w.shape[-1]
		d	  = int(self.Wxh_Left.shape[0]/3)
		e	  = self.E.shape[1] 
		C	  = self.Why_Left.shape[0]  # number of classes
		idx	= np.hstack((np.arange(0,d), np.arange(d,2*d))).astype(int) 
		
		# initialize
		Rx	   = np.zeros(self.x.shape)
		Rx_rev   = np.zeros(self.x.shape)
		
		Rh_Left  = np.zeros((T+1, d))
		Rc_Left  = np.zeros((T+1, d))
		Rg_Left  = np.zeros((T,   d)) # gate g only
		Rh_Right = np.zeros((T+1, d))
		Rc_Right = np.zeros((T+1, d))
		Rg_Right = np.zeros((T,   d)) # gate g only
		
		Rout_mask			= np.zeros((C))
		Rout_mask[LRP_class] = 1.0  
		
		s = self.s.copy()
		
		gates_Left = self.gates_Left.copy()
		gates_pre_Left = self.gates_pre_Left.copy()
		c_Left = self.c_Left.copy()
		h_Left = self.h_Left.copy()
		
		gates_Right = self.gates_Right.copy()
		gates_pre_Right = self.gates_pre_Right.copy()
		c_Right = self.c_Right.copy()
		h_Right = self.h_Right.copy()
		
		if self.mode == "dl":
			s -= self.ref.s

			gates_Left[:,2*d:] -= self.ref.gates_Left[:,2*d:]
			gates_pre_Left[:,2*d:] -= self.ref.gates_pre_Left[:,2*d:]
			c_Left -= self.ref.c_Left
			h_Left -= self.ref.h_Left
		
			gates_Right[:,2*d:] -= self.ref.gates_Right[:,2*d:]
			gates_pre_Right[:,2*d:] -= self.ref.gates_pre_Right[:,2*d:]
			c_Right -= self.ref.c_Right
			h_Right -= self.ref.h_Right
		
		# format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
		Rh_Left[T-1]  = lrp_linear(h_Left[T-1],  self.Why_Left.T , np.zeros((C)), s, s*Rout_mask, 2*d, eps, bias_factor, debug=False)
		Rh_Right[T-1] = lrp_linear(h_Right[T-1], self.Why_Right.T, np.zeros((C)), s, s*Rout_mask, 2*d, eps, bias_factor, debug=False)
		
		for t in reversed(range(T)):
			Rh_Left[t-1]  = lrp_linear(gates_Left[t,:d]*h_Left[t-1], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor, debug=False)
			Rg_Left[t]	= lrp_linear((1-gates_Left[t,:d])*gates_Left[t,2*d:], np.identity(d), np.zeros((d)), h_Left[t], Rh_Left[t], 2*d, eps, bias_factor, debug=False)
			Rx[t]		 = lrp_linear(self.x[t], self.Wxh_Left[2*d:].T, self.bxh_Left[2*d:]+self.bhh_Left[2*d:], gates_pre_Left[t,2*d:], Rg_Left[t], d+e, eps, bias_factor, debug=False)
			Rh_Left[t-1]  += lrp_linear(h_Left[t-1] * gates_Left[t,d:2*d], self.Whh_Left[2*d:].T, self.bxh_Left[2*d:]+self.bhh_Left[2*d:], gates_pre_Left[t,2*d:], Rg_Left[t], d+e, eps, bias_factor, debug=False)
			
			Rh_Right[t-1]  = lrp_linear(gates_Right[t,:d]*h_Right[t-1], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor, debug=False)
			Rg_Right[t]	= lrp_linear((1-gates_Right[t,:d])*gates_Right[t,2*d:], np.identity(d), np.zeros((d)), h_Right[t], Rh_Right[t], 2*d, eps, bias_factor, debug=False)
			Rx_rev[t]		 = lrp_linear(self.x_rev[t], self.Wxh_Right[2*d:].T, self.bxh_Right[2*d:]+self.bhh_Right[2*d:], gates_pre_Right[t,2*d:], Rg_Right[t], d+e, eps, bias_factor, debug=False)
			Rh_Right[t-1]  += lrp_linear(h_Right[t-1] * gates_Right[t,d:2*d], self.Whh_Right[2*d:].T, self.bxh_Right[2*d:]+self.bhh_Right[2*d:], gates_pre_Right[t,2*d:], Rg_Right[t], d+e, eps, bias_factor, debug=False)
		
		a, b, c = Rx, Rx_rev[::-1,:], Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum()

		summed = np.sum(a+b, axis = -1)
		relpeak = summed.argmax(-1) / len(summed)
		return a, b, c


if __name__ == "__main__":
	
	import _pickle
	import sys
	X = _pickle.load(open("../Inputs/yelp/X", "rb"))["dev"]
	Y = _pickle.load(open("../Inputs/yelp/Y", "rb"))["dev"]

	from keras.models import load_model
	from progressbar import ProgressBar

	if sys.argv[1] == "GRU":
		model = load_model("../Models/GRU_yelp.hdf5")
		weights = keras_to_weights(model, [0,1,2])
		lrp_model = BIGRU_np(weights, mode = "lrp")
	elif sys.argv[1] == "LSTM":
		model = load_model("../Models/LSTM_yelp.hdf5")
		weights = keras_to_weights(model, [0,2,1,3])
		lrp_model = BILSTM_np(weights, mode = "lrp")
	elif sys.argv[1] == "QGRU":
		model = load_model("../Models/QGRU_yelp.hdf5")
		weights = keras_to_weights(model, [0,1])
		lrp_model = BIQGRU_np(weights, mode = "lrp")
	elif sys.argv[1] == "QLSTM":
		model = load_model("../Models/QLSTM_yelp.hdf5")
		weights = keras_to_weights(model, [0,2,1,3])
		lrp_model = BIQLSTM_np(weights, mode = "lrp")
	elif sys.argv[1] == "CNN":
		model = load_model("../Models/CNN_yelp.hdf5")
		weights = keras_to_c2weights(model)
		lrp_model = BICNN_np(weights, mode = "lrp")
		
	corr = 0
	bar = ProgressBar()
	for x,y in bar(list(zip(X,Y))):
		prob = lrp_model.predict(x)
		corr += int(prob.argmax() == y)
	
	print("Accuracy", corr, "/", len(X), "=", corr / len(X))
