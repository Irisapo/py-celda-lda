      # To validate the celda result 

#######################  R code  ########################
## 1. In R, we simulate 100 cells with 100 genes coming from distinct 3 cell clusters. And save the counts matrix to be read into Python.

library(celda)
library(RcppCNPy)

sim_counts = simulateCells.celda_C( G = 100,K = 3)

npySave("sim_counts.npy", t(sim_counts$counts))  

# save as a cell by gene matrix 


#celda_cluster_assignments = celda( sim_counts$counts, model="celda_CG",sample.label = sim_counts$sample,K = 3,L = 3,max.iter = 25 )  


#######################  R code  ########################


###################   Python code   #######################
import matplotlib.pyplot as plt 
import sys, os
THEANO_FLAGS='floatX=float64,device=cpu'
import theano

import scipy 
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import seaborn as sns
from theano import shared
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams

import pymc3 as pm
from pymc3 import math as pmmath
from pymc3 import Dirichlet
from pymc3.distributions.transforms import t_stick_breaking




sim_counts = np.load("sim_counts.npy")
sim_counts.shape    # (447, 100)
gene_names = [ str(i) for i in list(range(1,sim_counts.shape[1]+1))]

# each row is a document, represented by 100-dimensional gene vector. 
plt.plot(sim_counts.T)
plt.show()


tf  = scipy.sparse.csr_matrix(sim_counts, shape = sim_counts.shape) # ndarray into matrix 
tf_celda = tf[:300, ]
docs_te = tf[300:, ]

n_tokens = np.sum(tf_celda[tf_celda.nonzero()]) 
print('Number of tokens = {}'.format(n_tokens))
print('Sparsity =  {}'.format (len(tf_celda.nonzero()[0]) / float(tf_celda.shape[0] * tf_celda.shape[1])))

# Log-likelihood of documents for LDA
## for a cell d sonsisting of gene w, the log-lihelihood of the LDA model with K topics is given as 
## where θd is the topic distribution for cell d and β is the gene distribution for the K topics. We define a funciton that returns a tensor of the log-likelihood of documents given θd and β. 

## ( β is a matrix parameter ) 
def logp_lda_doc(beta, theta):
    """Returns the log-likelihood function for given documents.
    K : number of topics in the model
    V : number of words (size of vocabulary)
    D : number of documents (in a mini-batch)
    Parameters
    ----------
    beta : tensor (K x V)
        Word distributions.
    theta : tensor (D x K)
        Topic distributions for documents.
    """
    def ll_docs_f(docs):
        dixs, vixs = docs.nonzero()
        vfreqs = docs[dixs, vixs]
        ll_docs = vfreqs * pmmath.logsumexp(
            tt.log(theta[dixs]) + tt.log(beta.T[vixs]), axis=1).ravel()
        # Per-word log-likelihood times num of tokens in the whole dataset
        return tt.sum(ll_docs) / (tt.sum(vfreqs)+1e-9) * n_tokens
    return ll_docs_f


## LDA model 
n_topics = 10   ## according to the result form celda_CG function 
n_genes = tf_celda.shape[1] 
n_cells = tf_celda.shape[0]

minibatch_size = 100

# defining minibatch
doc_t_minibatch = pm.Minibatch(tf_celda.toarray(), minibatch_size)

with pm.Model() as model:
    theta = Dirichlet('theta', a=pm.floatX((1.0 / n_topics) * np.ones((minibatch_size, n_topics))),
                      shape=(minibatch_size, n_topics), transform=t_stick_breaking(1e-9),
                      # do not forget scaling
                      total_size=tf_celda.shape[0])
    beta = Dirichlet('beta', a=pm.floatX((1.0 / n_topics) * np.ones((n_topics, n_genes))),
                     shape=(n_topics, n_genes), transform=t_stick_breaking(1e-9))
    # Note, that we devined likelihood with scaling, se here we need no additional `total_size` kwarg
    doc = pm.DensityDist('doc', logp_lda_doc(beta, theta), observed=doc_t_minibatch)



###### Auto-Encoding Variational Bayes
## Encoder
class LDAEncoder:
    """Encode (term-frequency) document vectors to variational means and (log-transformed) stds.
    """
    def __init__(self, n_genes, n_hidden, n_topics, p_corruption=0, random_seed=1):
        rng = np.random.RandomState(random_seed)
        self.n_genes = n_genes
        self.n_hidden = n_hidden
        self.n_topics = n_topics
        self.w0 = shared(0.01 * rng.randn(n_genes, n_hidden).ravel(), name='w0')
        self.b0 = shared(0.01 * rng.randn(n_hidden), name='b0')
        self.w1 = shared(0.01 * rng.randn(n_hidden, 2 * (n_topics - 1)).ravel(), name='w1')
        self.b1 = shared(0.01 * rng.randn(2 * (n_topics - 1)), name='b1')
        self.rng = MRG_RandomStreams(seed=random_seed)
        self.p_corruption = p_corruption
    def encode(self, xs):
        if 0 < self.p_corruption:
            dixs, vixs = xs.nonzero()
            mask = tt.set_subtensor(
                tt.zeros_like(xs)[dixs, vixs],
                self.rng.binomial(size=dixs.shape, n=1, p=1-self.p_corruption)
            )
            xs_ = xs * mask
        else:
            xs_ = xs
        w0 = self.w0.reshape((self.n_genes, self.n_hidden))
        w1 = self.w1.reshape((self.n_hidden, 2 * (self.n_topics - 1)))
        hs = tt.tanh(xs_.dot(w0) + self.b0)
        zs = hs.dot(w1) + self.b1
        zs_mean = zs[:, :(self.n_topics - 1)]
        zs_std = zs[:, (self.n_topics - 1):]
        return zs_mean, zs_std
    def get_params(self):
        return [self.w0, self.b0, self.w1, self.b1]




encoder = LDAEncoder(n_genes = n_genes, n_hidden=33, n_topics=n_topics, p_corruption=0.0)
local_RVs = OrderedDict([(theta, encoder.encode(doc_t_minibatch))])



encoder_params = encoder.get_params() 


#### model checking for the theta using AEVB
with model:
    t0 = time()
    approx1 = pm.fit(6000, method='advi',
                 local_rv=local_RVs,
                 more_obj_params=encoder_params,
                 # https://arxiv.org/pdf/1705.08292.pdf
                 # sgd(with/without momentum) seems to be good choice for high dimensional problems
                 obj_optimizer=pm.sgd,
                 # but your gradients will explode here
                 total_grad_norm_constraint=1000.)
    print("done in %0.3fs." % (time() - t0))

plt.plot(approx1.hist[10:])
plt.show()


#Extraction of characteristic words of topics based on posterior samples
theano.config.compute_test_value = 'raise'
n_docs_te = docs_te.shape[0]
doc_t = shared(docs_te.toarray(), name='doc_t')

with pm.Model() as model:
    theta = Dirichlet('theta', a=pm.floatX((1.0 / n_topics) * np.ones((n_docs_te, n_topics))),
                      shape=(n_docs_te, n_topics), transform=t_stick_breaking(1e-9))
    beta = Dirichlet('beta', a=pm.floatX((1.0 / n_genes) * np.ones((n_topics, n_genes))),
                     shape=(n_topics, n_genes), transform=t_stick_breaking(1e-9))
    doc = pm.DensityDist('doc', logp_lda_doc(beta, theta), observed=doc_t)
    encoder.p_corruption = 0
    local_RVs = OrderedDict([(theta, encoder.encode(doc_t))])
    approx = pm.MeanField(local_rv=local_RVs)
    approx.shared_params = approx1.shared_params



topic_gene = {}
def print_top_genes(beta, gene_names, n_top_genes=100):
    for i in range(len(beta)):
        print(("Topic #%d: " % i) + " ".join([gene_names[j]
            for j in beta[i].argsort()[:-n_top_genes - 1:-1]]))
        topic_gene["Topic" + str(i)] = [j for j in beta[i].argsort()[:-n_top_genes - 1:-1]]


with model:
    samples = pm.sample_approx(approx, draws=122)
    beta_pymc3 = samples['beta'].mean(axis=0)


print_top_genes(beta_pymc3, gene_names)



arr = np.array([topic_gene[key] for key in list(topic_gene.keys())]).T
np.savetxt("topic_gene.csv", arr, delimiter = ",")
np.savetxt("beta.csv", beta_pymc3, delimiter = "," )

########################## R code for comparing result in visualization ###################################

#library(reshape2)

#topic_gene.py <- read.csv("topic_gene.csv", header = FALSE)
#colnames(topic_gene.py) <- paste0("Topic", seq(0,9))

#topic_gene.py.melt <- melt(topic_gene.py)
#colnames(topic_gene.py.melt) <- c("Topic", "gene") 
#topic_gene.py.melt$plot = 2


#topic_gene.celda = data.frame(topic = celda_cluster_assignments$res.list[[1]]$y, 
#                              gene = seq(1,length(celda_cluster_assignments$res.list[[1]]$y)) )
#topic_gene.celda$plot = 1


beta <- read.csv("beta.csv", header=FALSE)

apply(beta,2,which.max)


table(apply(beta,2,which.max), celda_cluster_assignments$res.list[[1]]$y ) 




















