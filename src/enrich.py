import pandas as pd
import argparse
import time
from scipy.stats import chi2_contingency
import numpy as np
from scipy.special import softmax
import os
import sys

np.set_printoptions(precision=4, linewidth=200)

def title():
    print('**********************************************************************')
    print('* SparsePro for testing functional enrichment of annotations         *')
    print('* Version 1.0.1                                                      *')
    print('* (C) Wenmin Zhang (wenmin.zhang@mail.mcgill.ca)                     *')
    print('**********************************************************************')
    print()

def get_sig_enrich(A,all_PIP):
    
    W = np.zeros(A.shape[1])
    W_se = np.zeros(A.shape[1])
    eps = 1000
    tot = all_PIP.sum()
    
    for ite in range(20):
        W_old = W.copy()
        for i in range(A.shape[1]):
            idxall = [x for x in range(A.shape[1])]
            idxall.remove(i)
            k = softmax(np.dot(A[:,idxall],W[idxall]))
            kr = k[A[:,i]==1].sum()
            r = all_PIP[np.where(A[:,i])[0]].sum()/tot
            W_new = np.log((1-kr) * r / (1-r) / (kr))
            W[i] = W_new
            W_se_new = np.sqrt(1/(r*tot)+1/((1-r)*tot)-1/(kr*A.shape[0])-1/((1-kr)*A.shape[0]))
            W_se[i] = W_se_new
        eps = ((W - W_old)**2).sum()
        #print("iteration {} with diff {}".format(ite,eps))
        if eps < 1e-2:
            #print("converged")
            break
            
    return W,W_se

parser = argparse.ArgumentParser(description='SparsePro Enrich Commands:')
parser.add_argument('--save', type=str, default=None, help='path to save result', required=True)
parser.add_argument('--prefix', type=str, default=None, help='prefix for result files', required=True)
parser.add_argument('--anno', type=str, default=None, help='path to annotation file',required=True)
parser.add_argument('--pip', type=str, default=None, help='path to pip file',required=True)
parser.add_argument('--pthres', type=float, default=None, help='p value threshold for enrichment',required=True)
args = parser.parse_args()

title()

if not os.path.exists(args.save):
    os.makedirs(args.save)

allPIP =  pd.read_csv(args.pip,sep='\s+',index_col=0,header=None)
anno = pd.read_csv(args.anno,sep='\s+',index_col=0)
print("Annotation file Loaded at {}".format(time.strftime("%Y-%m-%d %H:%M")))
paidx = anno.index.intersection(allPIP.index)
print("There are {} variants with {} annotations and among them {} variants have PIP esitmates".format(anno.shape[0],anno.shape[1],len(paidx)))
print()
allPIP = allPIP.loc[paidx]
anno = anno.loc[paidx]

Wsep = {}

for k in anno.columns:
    P = len(anno[k])
    A = (anno[k]).sum()
    K = allPIP.values.sum()
    M = allPIP.loc[anno[k]==1].values.sum()
    
    obs = np.array([[K-M,P-A-K+M],[M,A-M]])
    g, p, dof, expctd = chi2_contingency(obs, lambda_="log-likelihood")
    W = np.log(M*(P-A)/A/(K-M))
    W_se = np.sqrt(1/M + 1/(K-M) - 1/A - 1/(P-A))
    Wsep[k] = [W,W_se,p]

df_Wsep = pd.DataFrame(Wsep).round(4)
df_Wsep.index = ['W','se','p']
df_Wsep.index.name = 'Wsep'
df_Wsep = df_Wsep.transpose()

print("Univariate testing finished at {}. Saving result to {}.wsep file...".format(time.strftime("%Y-%m-%d %H:%M"),args.prefix))
print()

df_Wsep.to_csv(os.path.join(args.save,'{}.wsep'.format(args.prefix)),sep="\t")

sigidx = [i for i in range(anno.shape[1]) if df_Wsep.p[i]<args.pthres]

if len(sigidx)==0:
    sys.exit("None of the {} annotations is significantly enriched at p-value threshold {}. Existing...".format(anno.shape[1], args.pthres))
else:
    print("{} annotations are deemed significantly enriched at {} p-value threshold and used to update priors. Saving result to {}.W{} file...".format(len(sigidx),args.pthres,args.prefix,args.pthres))

sigANNOT = anno.values[:,sigidx]

W_sig,W_se_sig = get_sig_enrich(sigANNOT, allPIP.values)

df_W_sig = pd.DataFrame({'ANNO':anno.columns[sigidx],'W_sig':W_sig, 'W_se_sig':W_se_sig, 'sigidx':sigidx})
#print(df_W_sig)
df_W_sig.to_csv(os.path.join(args.save,'{}.W{}'.format(args.prefix,args.pthres)),sep='\t',index=False)
