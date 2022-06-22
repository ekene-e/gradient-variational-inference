import pandas as pd
import argparse
import os
import time
import numpy as np
import pickle
from scipy.special import softmax
import scipy.sparse as sparse

np.set_printoptions(precision=4, linewidth=200)

def title():
    print('**********************************************************************')
    print('* SparsePro+ for annotated fine-mapping                              *')
    print('* Version 1.0.1                                                      *')
    print('* (C) Wenmin Zhang (wenmin.zhang@mail.mcgill.ca)                     *')
    print('**********************************************************************')
    print()

def get_XX_XtX_ytX(LD,beta,se,var_Y):
    '''get sufficient statistics from summary statistics'''
    XX = var_Y/(se**2)
    XtX = LD * var_Y / (np.dot(se.reshape(-1,1),se.reshape(1,-1)))
    ytX = XX * beta
    return XX, XtX, ytX

#obtain from https://storage.googleapis.com/broad-alkesgroup-public/UKBB_LD/readme_ld.txt
def load_ld_npz(ld_prefix):
    
    #load the SNPs metadata
    gz_file = '%s.gz'%(ld_prefix)
    df_ld_snps = pd.read_table(gz_file, sep='\s+')
    df_ld_snps.rename(columns={'rsid':'SNP', 'chromosome':'CHR', 'position':'BP', 'allele1':'A1', 'allele2':'A2'}, inplace=True, errors='ignore')
    assert 'SNP' in df_ld_snps.columns
    assert 'CHR' in df_ld_snps.columns
    assert 'BP' in df_ld_snps.columns
    assert 'A1' in df_ld_snps.columns
    assert 'A2' in df_ld_snps.columns
    df_ld_snps.index = df_ld_snps['CHR'].astype(str) + '.' + df_ld_snps['BP'].astype(str) + '.' + df_ld_snps['A1'] + '.' + df_ld_snps['A2']
        
    #load the LD matrix
    npz_file = '%s.npz'%(ld_prefix)
    try: 
        R = sparse.load_npz(npz_file).toarray()
        R += R.T
    except ValueError:
        raise IOError('Corrupt file: %s'%(npz_file))
    df_R = pd.DataFrame(R, index=df_ld_snps.index, columns=df_ld_snps.index)
    return df_R, df_ld_snps

class SparsePro(object):
    
    def __init__(self,P,K,XX,var_Y,h2):
        '''initialize and set hyperparameters'''
        self.p = P
        self.k = K
        self.gamma = np.zeros((self.p,self.k))
        self.beta_mu = np.zeros((self.p,self.k))
        self.beta_prior_tau = np.tile((1.0 / var_Y * h2 * np.array([k+1 for k in range(self.k)])),(self.p,1))
        self.y_tau = 1.0 / (var_Y * (1-h2))
        self.prior_pi = np.ones((self.p,)) * (1/self.p)
        self.beta_post_tau = np.tile(XX.reshape(-1,1),(1,self.k)) * self.y_tau + self.beta_prior_tau
        
    def infer_q_beta(self,XX,ytX,XtX,LD):
        '''perform variational updates'''
        for k in range(self.k):
            idxall = [x for x in range(self.k)]
            idxall.remove(k)
            beta_all_k = (self.gamma[:,idxall] * self.beta_mu[:,idxall]).sum(axis=1)
            self.beta_mu[:,k] = (ytX-np.dot(beta_all_k, XtX))/self.beta_post_tau[:,k] * self.y_tau
            u = -0.5*np.log(self.beta_post_tau[:,k]) + np.log(self.prior_pi.transpose()) + 0.5 * self.beta_mu[:,k]**2 * self.beta_post_tau[:,k]
            self.gamma[:,k] = softmax(u)
            #maxid = np.argmax(u)
            #self.gamma[abs(LD[maxid])<0.05,k]= 0.0

    def get_elbo(self):
        
        beta_all = (self.gamma * self.beta_mu).sum(axis=1)
        ll1 = self.y_tau * np.dot(beta_all,ytX)
        ll2 = - 0.5 * self.y_tau * ((((self.gamma * self.beta_mu**2).sum(axis=1) * XX).sum()))
        W = self.gamma * self.beta_mu
        WtRW = np.dot(np.dot(W.transpose(),XtX),W)
        ll3 = - 0.5 * self.y_tau * ( WtRW.sum() - np.diag(WtRW).sum())
        ll = ll1 + ll2 + ll3
        betaterm1 = -0.5 * (self.beta_prior_tau * self.gamma * (self.beta_mu**2)).sum()
        gammaterm1 = (self.gamma * np.tile(self.prior_pi.reshape(-1,1),(1,self.k))).sum()
        gammaterm2 = (self.gamma[self.gamma!=0] * np.log(self.gamma[self.gamma!=0])).sum()
        mkl = betaterm1 + gammaterm1 - gammaterm2
        elbo = ll + mkl
        
        return ll,mkl,elbo
       
    def get_PIP(self):
        
        return np.max((self.gamma),axis=1)
        
    def update_pi(self, new_pi):
        
        self.prior_pi = new_pi

    def get_effect_dict(self):
        
        numidx = (self.gamma>0.1).sum(axis=0)
        matidx = np.argsort(-self.gamma, axis=0)
        return {i:matidx[0:numidx[i],i].tolist() for i in range(self.k) if numidx[i]>0}
    
    def get_effect_num_dict(self):
        
        gamma = np.round(self.gamma,4)
        beta_mu = np.round(self.beta_mu,4)
        effect = self.get_effect_dict()
        eff_gamma = {i:gamma[effect[i],i].tolist() for i in effect}
        eff_mu = {i:beta_mu[effect[i],i].tolist() for i in effect}
        
        return eff_gamma, eff_mu
    
    def train(self,XX,ytX,XtX,LD,maxite=50,eps=0.01,verbose=False,loss=0.0):
        
        for ite in range(maxite):
            self.infer_q_beta(XX, ytX, XtX, LD)
            ll, mkl, elbo = self.get_elbo()
            if verbose:
                print('*'*70)
                print('Iteration-->{} . Likelihood: {:.2f} . KL: {:.2f} . ELBO: {:.2f}'.format(ite, ll, mkl, elbo))
            if abs(elbo-loss)<eps:
                break
            loss = elbo

parser = argparse.ArgumentParser(description='SparsePro+ Commands:')
parser.add_argument('--ss', type=str, default=None, help='path to summary stats', required=True)
parser.add_argument('--var_Y', type=float, default=None, help='GWAS trait variance', required=True)
parser.add_argument('--N', type=int, default=None, help='GWAS sample size', required=True)
parser.add_argument('--K', type=int, default=None, help='largest number of effect', required=True)
parser.add_argument('--LDdir', type=str, default=None, help='path to LD files', required=True)
parser.add_argument('--LDlst', type=str, default=None, help='path to LD list', required=True)
parser.add_argument('--save', type=str, default=None, help='path to save result', required=True)
parser.add_argument('--prefix', type=str, default=None, help='prefix for result files', required=True)
parser.add_argument('--anno', type=str, default=None, help='path to annotation file')
parser.add_argument('--W', type=str, default=None, help='path to enriched file',required=True)
parser.add_argument("--verbose", action="store_true", help='options for displaying more information')
parser.add_argument("--ukb", action="store_true", help='options for using precomputed UK Biobank ld files from PolyFun')

args = parser.parse_args()

title()

if not os.path.exists(args.save):
    os.makedirs(args.save)

ss = pd.read_csv(args.ss,sep="\s+",dtype={'SNP':str,'BETA':float,'SE':float},index_col=0)
print("summary statistics loaded at {}".format(time.strftime("%Y-%m-%d %H:%M")))
anno = pd.read_csv(args.anno,sep='\s+',index_col=0)
print("Annotation file Loaded at {}".format(time.strftime("%Y-%m-%d %H:%M")))
paidx = anno.index.intersection(ss.index)
print("There are {} variants with {} annotations and among them {} variants have summary statistics".format(anno.shape[0],anno.shape[1],len(paidx)))
print()

ldlists=pd.read_csv(args.LDlst,sep='\s+',dtype={'ld':str,'start':int,'end':int})
print("LD list with {} LD blocks loaded\n".format(len(ldlists)))

W_sig = pd.read_csv(args.W,sep="\s+")
sigidx = W_sig['sigidx'].tolist()
W_new = W_sig['W_sig'].values

apip = []
apip_name=[]
acs = []
acs_pip = []
acs_eff = []
atl = []


for i in range(len(ldlists)):
    ld = ldlists['ld'][i]
    start = ldlists['start'][i]
    end = ldlists['end'][i]
    
    objfile = os.path.join(args.save,'{}.obj'.format(ld))
    if os.path.exists(objfile):
        open_file = open(objfile,'rb')
        h2_hess,var_b,model,elbo = pickle.load(open_file)
        open_file.close()
    else:
        print("No intermediate file saved, skipping...")
        continue
    
    if args.ukb:
        ldfile = ld.replace('.npz','')
        df_R, df_ld_snps = load_ld_npz(os.path.join(args.LDdir,ldfile))
        idx = df_R.index.intersection(ss.index)
        LD = df_R.loc[idx,idx]
    else:
        LD = pd.read_csv(os.path.join(args.LDdir,ld),sep='\t',index_col=0)
        idx = LD.index.intersection(ss.index)
        
    if len(idx)<20:
        print("Not enough variants found, skipping")
        continue

    pos = [int(i.split('.')[1]) for i in idx]
    
    beta = ss.loc[idx,'BETA'].values
    se = ss.loc[idx,'SE'].values
    XX, XtX, ytX = get_XX_XtX_ytX(LD.values,beta,se,args.var_Y)
    
    print("{} variants loaded from {} with {} variants having matched summary statistics explaining {:2.2%} of trait heritability \n".format(LD.shape[1], ld, len(idx), h2_hess))
    
    effidx = [i for i in range(len(idx)) if ((pos[i] >= start) & (pos[i] < end))]
    effnum = len(effidx)
    
    print('{} variants in the range of {} to {}'.format(effnum, start, end))
    if effnum <=20:
        print('Not enough effective variants, skipping')
        continue
    
    ianno = anno.loc[idx]
    ANN = ianno.values[:,sigidx]
    new_pi= softmax(np.dot(ANN,W_new))
    
    model.update_pi(new_pi)
    model.train(XX, ytX, XtX, LD.values,verbose=args.verbose,loss=elbo)
    
    mcs = model.get_effect_dict()
    eff_gamma, eff_mu = model.get_effect_num_dict()
    
    pip_vec = model.get_PIP().round(4)
    apip.extend([pip_vec[i] for i in effidx])
    apip_name.extend([idx[i] for i in effidx])
    
    if len(mcs)==0:
        print("No effect detected")
        print()
        continue

    print("Detected k = {}".format(list(mcs)[-1]+1))
    print()
    for i in mcs:
        if mcs[i][0] in effidx:
            atl.append(idx[mcs[i][0]])
            mcs_idx = [idx[j] for j in mcs[i]]
            print('The {}-th effect contains effective variants:'.format(i))
            print('causal variants: {}'.format(mcs_idx))
            print('posterior inclusion probabilities: {}'.format(eff_gamma[i]))
            print('posterior causal effect size: {}'.format(eff_mu[i])) 
            print()
            acs.append(mcs_idx)
            acs_pip.append(eff_gamma[i])
            acs_eff.append(eff_mu[i])
    

allPIP = pd.DataFrame({"idx":apip_name,"pip":apip})  
allPIP.to_csv(os.path.join(args.save,"{}.apip".format(args.prefix)),sep='\t',header=False,index=False)   
allcs = pd.DataFrame({"cs":acs,"pip":acs_pip,"beta":acs_eff})
allcs.to_csv(os.path.join(args.save,"{}.acs".format(args.prefix)),sep='\t',header=True,index=False)
pd.Series(atl,dtype='str').to_csv(os.path.join(args.save,"{}.atl".format(args.prefix)),sep='\t',header=False,index=False)
print("Annotated fine-mapping finished at {}. Writing all PIPs to {}.apip; all credible sets to {}.acs; all top snps in each effect to {}.atl ...".format(time.strftime("%Y-%m-%d %H:%M"),args.prefix,args.prefix,args.prefix))