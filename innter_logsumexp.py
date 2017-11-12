


x2 = tt.matrix("x2")
y2 = tt.matrix("y2")
n = tt.lscalar("n")
ind = tt.lvector("ind")  # int64



results2, updates2 = theano.scan(lambda x2,y2,n,ind: 
                                 # n is the length of ind(ex);  x2: phi ;  y2: psi
                                 pmmath.logsumexp(tt.log(tt.tile(x2, (n,1))) + tt.log(y2.T[ind]), axis=1).ravel(), 
                                 sequences=x2, 
                                 non_sequences=[y2,n,ind])
topic_gene_loop = theano.function(inputs=[x2,y2,n,ind], outputs=results2)  # but output is 3 dimension... !!!!!!! 
# so what to do to make it 2 dimensions  



# ex:
counts = np.array([[1, 2, 3, 0, 0],[2, 0, 3, 5, 9],[9, 0, 0, 1, 2]])
theta = np.array([[ 0.2,  0.8],[ 0.5,  0.5],[ 0.6,  0.4]])
phi = np.array([[ 0.2,  0.2,  0.3,  0.3],[ 0.1,  0.3,  0.4,  0.2]])
psi = np.array([[ 0.2,  0.2,  0.2,  0.2,  0.2],[ 0.2,  0.2,  0.2,  0.2,  0.2],[ 0.2,  0.2,  0.2,  0.2,  0.2],[ 0.2,  0.2,  0.2,  0.2,  0.2]])



dixs, vixs = counts.nonzeros()
N = len(dixs)   # N = len(dixs.eval())  ???
dixs1 = dixs.astype(np.int32)


topic_gene = topic_gene_loop(phi, psi, N, dixs1)



results2, updates2 = theano.scan(lambda x2,psi,N_share,ind: 
                                 # n is the length of ind(ex);  x2: phi ;  y2: psi
                                 pmmath.logsumexp(tt.log(tt.tile(x2, (N_share,1))) + tt.log(psi.T[ind]), axis=1).ravel(), 
                                 sequences=x2, 
                                 non_sequences=[psi,N_share,ind])