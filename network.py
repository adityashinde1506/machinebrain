import numpy
import theano
import theano.tensor as T
from layers import *

class Network:

	def __init__(self,layers):
		self.layers=layers


class FeedForwardNetwork(Network):

	def __init__(self,layers):
		Network.__init__(self,layers)
		inp=T.dmatrix()
		params=[]
		for layer in self.layers:
			params+=layer.params
		self.output=theano.function([self.layers[0].inp],self.layers[-1].output)
		labels=T.dmatrix()
		reg_lambda=T.dscalar()
		alpha=T.dscalar()
		cost=T.nnet.binary_crossentropy(self.layers[-1].output,labels).mean()
		for layer in self.layers:
			cost+=(reg_lambda*layer.reg)
		updates=[(param,param-(alpha*T.grad(cost,param))) for param in params]
		self.backprop=theano.function([self.layers[0].inp,labels,alpha,reg_lambda],cost,updates=updates)

	def predict(self,X):
		return self.output(X)
	
	def train(self,X,y,alpha=0.01,reg_lambda=0.00001,convergence=None,epochs=None):
		i=0
		if epochs==None:
			cost=1
			while cost>convergence:
				i+=1
				cost=self.backprop(X,y,alpha,reg_lambda)
				print "Epoch> "+str(i)+" Cost: "+str(cost)
		else:
			while epochs:
				i+=1
				print "Epoch> "+str(i)+" Cost: "+str(self.backprop(X,y,alpha,reg_lambda))
				epochs-=1

class AE(Network):

	def __init__(self,layers):
		Network.__init__(self,layers)
		inp=T.dmatrix()
		params=self.layers.params
		self.output=theano.function([self.layers.inp],self.layers.output)
		self.recovered=theano.function([self.layers.inp],self.layers.recovered)
		alpha=T.dscalar()
		cost=((self.layers.inp-self.layers.recovered)**2).mean()
		updates=[(param,param-(alpha*T.grad(cost,param))) for param in params]
		self.backprop=theano.function([self.layers.inp,alpha],cost,updates=updates)

	def predict(self,X):
		return self.output(X)
	
	def train(self,X,alpha=0.001,convergence=None,epochs=None):
		i=0
		if epochs==None:
			cost=1
			while cost>convergence:
				i+=1
				cost=self.backprop(X,alpha)
				print "Epoch> "+str(i)+" Cost: "+str(cost)
		else:
			while epochs:
				i+=1
				print "Epoch> "+str(i)+" Cost: "+str(self.backprop(X,alpha))
				epochs-=1



if __name__=='__main__':
	X=numpy.array([[0.,0.,0.,1.,0.,0.],[0.,1.,1.,0.,0.,0.],[1.,0.,0.,0.,1.,1.],[1.,1.,0.,0.,1.,1.]])
	layer0=Autoencoder(6,5)
	net=AE(layer0)
	net.train(X,epochs=100000)


