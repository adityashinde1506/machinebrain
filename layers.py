import numpy
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as convop
from theano.tensor.signal import downsample


theano.config.exception_verbosity='high'
class Layer:
	
	def __init__(self,n_in,n_out,W=None,b=None,non_linearity=None):
		self.n_in=n_in
		self.n_out=n_out
		if W==None:
			if non_linearity=='tanh':
				self.W=theano.shared(numpy.random.uniform(low=-numpy.sqrt(6./(n_in+n_out)),high=numpy.sqrt(6./(n_in+n_out)),size=(n_out,n_in)),borrow=True)
			else:
				self.W=theano.shared(numpy.random.uniform(low=-4*numpy.sqrt(6./(n_in+n_out)),high=4*numpy.sqrt(6./(n_in+n_out)),size=(n_out,n_in)),borrow=True)			
		else:
			self.W=W
		if b==None:
			self.b=theano.shared(numpy.random.random((n_out)),borrow=True)
		else:
			self.b=b
		self.params=[self.W,self.b]
		self.non_lins={'sigmoid':T.nnet.sigmoid,'softmax':T.nnet.softmax,'tanh':T.tanh}
		self.reg=T.sum(self.W**2)+T.sum(self.b**2)

class InputLayer(Layer):

	def __init__(self,inp_vector=T.dmatrix(),W=None,b=None):
		self.W=theano.shared(0.)
		self.b=theano.shared(0.)
		Layer.__init__(self,0,0,self.W,self.b)
		self.params=[]
		self.inp=inp_vector
		self.output=self.inp

class HiddenLayer(Layer):

	def __init__(self,n_in,n_out,inp_vector=T.dmatrix(),non_linearity='sigmoid',W=None,b=None):
		Layer.__init__(self,n_in,n_out,W,b)
		self.inp=inp_vector
		self.output=self.non_lins[non_linearity](T.dot(inp_vector,self.W.transpose())+self.b)
		

class OutputLayer(Layer):

	def __init__(self,n_in,n_out,inp_vector=T.dmatrix(),non_linearity='softmax',W=None,b=None):
		Layer.__init__(self,n_in,n_out,W,b)
		self.inp=inp_vector
		out=self.non_lins[non_linearity](inp_vector.dot(self.W.transpose())+self.b)
		self.output=out

class ConvolutionLayer(Layer):

	"""Input image should be a 4D tensor with dimensions (n_images,channels,height,width) (This can be overwritten by passing an N dimensional tensor to inp_vector.)
		W_shape should have as many dimensions as the input image, for 4D tensors it should be (output feature_maps,input feature_maps,height,width)"""

	def __init__(self,W_shape,b_shape,image_shape,inp_vector=T.tensor4(),maxpool=(2,2),non_linearity='tanh',W=None,b=None,flatten=False,batch=1):
		bound=W_shape[1]*W_shape[2]*W_shape[3]
		W=theano.shared(numpy.random.uniform(low=-1./bound,high=1./bound,size=W_shape),borrow=True)
		b=theano.shared(numpy.random.random(b_shape))
		Layer.__init__(self,0,0,W,b)
		self.inp=inp_vector
		out=convop.conv2d(input=inp_vector,filters=self.W,filter_shape=W_shape,image_shape=image_shape)
		self.output=self.non_lins[non_linearity](downsample.max_pool_2d(out,maxpool,ignore_border=True)+self.b.dimshuffle('x',0,'x','x'))
		if flatten:
			self.output=self.output.flatten(batch)
		else:
			self.output=self.output
	
			
		


		
