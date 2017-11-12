
import numpy as np
import scipy

from collections import OrderedDict

import pymc3 as pm
from pymc3 import * 
from pymc3 import math as pmmath
from pymc3.distributions.transforms import t_stick_breaking

THEANO_FLAGS='floatX=float64,device=cpu'
import theano
import theano.tensor as tt
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStreams


sim_counts = np.load("sim_counts.npy")
z_celda = np.load("celda_cell_label.npy")

counts_share = theano.shared(sim_counts)


n_topics = 5 
n_genes = sim_counts.shape[1]

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
        return tt.sum(ll_docs) 
    return ll_docs_f
    

with pm.Model() as lda_model:
    theta = Dirichlet('theta', a=pm.floatX(1.0/n_topics) * np.ones((sim_counts.shape[0], n_topics)), shape = (sim_counts.shape[0],n_topics), transform=t_stick_breaking(1e-9) )
    beta = Dirichlet('beta', a = pm.floatX(1.0/n_topics) * np.ones((n_topics, sim_counts.shape[1])), shape= (n_topics, sim_counts.shape[1]), transform=t_stick_breaking(1e-9) )
    doc = pm.DensityDist('doc', logp_lda_doc(beta, theta), observed = sim_counts)
    
    
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
local_RVs = OrderedDict([(theta, encoder.encode(counts_share))])

encoder_params = encoder.get_params()  



with lda_model:
    approx1 = pm.fit(6000, method='advi',
                 local_rv=local_RVs,
                 more_obj_params=encoder_params,
                 # https://arxiv.org/pdf/1705.08292.pdf
                 # sgd(with/without momentum) seems to be good choice for high dimensional problems
                 obj_optimizer=pm.sgd,
                 # but your gradients will explode here
                 total_grad_norm_constraint=1000)
    samples = pm.sample_approx(approx1, draws=100)
    beta_pymc3 = samples['beta'].mean(axis=0)
    theta_pymc3 = samples['theta'].mean(axis=0)


plt.plot(approx1.hist[10:]) 
plt.show()


## get label for each cell
z_pymc3  = theta_pymc3.argmax(axis=1)

pd.DataFrame({"celda":z_celda, "pymc3":z_pymc3}).groupby(['celda','pymc3']).size()

## 3:21 minutes to run the AEVB with ADVI to get the posterior Dis 
## dataset 507 cells x 100 genes ( 5 topics ) from celda  


    
