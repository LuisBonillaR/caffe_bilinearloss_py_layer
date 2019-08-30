from __future__ import print_function
import caffe
import numpy as np

class BilinearLossLayer(caffe.Layer):
    def softmax(self, features):
        '''
        Calculates the softmax and logsoftmax of the features on a stable way.
        "Features" includes the whole batch and (log)softmax is calculated
        event(row)-wise.
        '''
        features -= np.max(features, axis=1)[:, np.newaxis]
        exps = np.exp(features)
        
        logsoftmax = features - np.log(exps.sum(axis=1)[:, np.newaxis])
        softmax = np.exp(logsoftmax)

        return softmax, logsoftmax


    def loss(self, label, pred, logpred, pen_mat, alpha=.5, log=False):
        '''
        This function calculates cross entropy loss plus (log)bilinear loss.
        Also we calculate the loss function gradient per event. It is needed to
        give the label of the event as well as the prediction vector (usually
        output from softmax) and the log of the prediction. This also needs as
        input the desired penalty matrix and the parameter alpha. We can choose
        whether to use bilinear loss or logbilinear loss.
        '''
        # this must work regardless the number of labels
        labels = pred.shape[1]
        beta = 1-alpha
        
        # we need to make sure label type is int
        label = np.transpose(label.astype(np.int))
        
        # prepare array for advanced indexing
        batch_row = np.arange(label.shape[1]).reshape(label.shape)
        
        # cross entropy, loss and diff
        LCE = -logpred[batch_row, label].T
        diff_CE = np.copy(pred)
        diff_CE[batch_row, label] -= 1
        
        # x will be used to calculate the grad of the bilinear loss
        x = np.zeros_like(pred)
        LB = np.zeros_like(LCE)
        diff_B = np.zeros_like(pred)
        
        # (log)bilinear, loss and diff
        if log and alpha != 0.:
            LB = -np.multiply(pen_mat[label],np.log(1-pred+1e-10)).sum(axis=2).T
            for l in range(0,labels):      # diff_B components
                x = -pred/(1-pred+1e-10)*(pred[:,l][:, np.newaxis])
                x[:,l] = pred[:,l]
                diff_B[:,l] = np.multiply(pen_mat[label], x).sum(axis=2)
        
        if not log and alpha != 0.:
            LB = np.multiply(pen_mat[label], pred).sum(axis=2).T
            for l in range(0,labels):      # diff_B components
                x = -pred*(pred[:,l][:, np.newaxis])
                x[:,l] = pred[:,l]*(1-pred[:,l])
                diff_B[:,l] = np.multiply(pen_mat[label], x).sum(axis=2)
        
        return beta*LCE + alpha*LB, beta*diff_CE + alpha*diff_B


    def setup(self, bottom, top):
        # check for all inputs
        if len(bottom) != 2:
            raise Exception("Need two inputs (features and labels) to "
                "compute (log)bilinear loss.")
        
        n_labels = bottom[0].data.shape[1] #len(bottom[0].data[1])
        params = eval(self.param_str)
        
        self.alpha = params.get('alpha', 0.5)
        self.log = params.get('log', False)
        self.pen_mat = params.get('pen_mat', np.ones((n_labels, n_labels)))
        self.pen_mat -= self.pen_mat*np.eye(self.pen_mat.shape[0])
        self.pen_mat /= self.pen_mat.sum(axis=1)
#        self.pen_mat /= np.amax(self.pen_mat)


    def reshape(self, bottom, top):
        # check input dimensions match between the predictionss and labels
        if bottom[0].shape[0] != bottom[1].shape[0]:
            raise Exception("Inputs must have the same dimension.")
        
        # layer output will be an averaged scalar loss
        top[0].reshape(1)
        top[0].data[...] = 0


    def forward(self, bottom, top):
        features = bottom[0].data
        labels = bottom[1].data 
        
        # softmax and logsoftmax
        pred, logpred = self.softmax(features)
        
        # len(label) is equal to the batch size
        loss, grad = self.loss(labels, pred, logpred,
                            self.pen_mat, self.alpha, log=self.log)                           
        
        top[0].data[...] = np.mean(loss)
        
        self.diff = grad / labels.shape[0] #len(labels)
        
        if np.isnan(top[0].data):
            raise Exception("The loss is NaN and cannot continue")


    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.diff
        
