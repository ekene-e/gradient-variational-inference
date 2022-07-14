from unicodedata import decimal
import pandas as pd
import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import scipy.sparse as sparse

np.set_printoptions(precision=4, linewidth=200)

def title():
    print('**********************************************************************')
    print('* SparsePro for efficient genome-wide fine-mapping                   *')
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

#unstandardized Heritability Estimate from Summary Staistics (HESS) extended from Shi et al.,2016
def get_HESS_h2_SS(XtX,XX,LD,beta,se,N,var_Y,LDthres=0.1):
    '''calculate local heritabilities'''
    idx_retain = []
    idx_exclude = [i for i in range(len(beta))]
    zscore=np.abs(beta/se)
    while len(idx_exclude)>0:
        maxid = idx_exclude[np.argmax(zscore[idx_exclude])]
        idx_retain.append(maxid)
        idx_exclude = [i for i in idx_exclude if i not in np.where(LD[maxid,:]>LDthres)[0]]
    Indidx = np.sort(idx_retain)
    #obtain independent signals
    P = len(Indidx)
    XtX_id = XtX[np.ix_(Indidx,Indidx)]
    R_inv = np.linalg.inv(XtX_id)
    vec_id = XX[Indidx] * beta[Indidx]
    h2_hess = (np.dot(np.dot(vec_id.transpose(),R_inv),vec_id)-var_Y*P)/(var_Y*(N-P))
    var_b = np.median(beta[Indidx]**2)
    
    if h2_hess<0.0001:
        h2_hess = 0.0001
    if h2_hess>0.9:
        h2_hess = 0.9
    return h2_hess,var_b

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

def make_tensors(*args):
    tensor_list = []
    for array in args:
        tensor_list.append(torch.tensor(array, dtype=torch.float32))
    return tensor_list

class SparsePro(nn.Module):

    def __init__(self,P,K,XX,var_Y,h2,var_b):
        '''initialize as torch tensors and set hyperparameters'''
        super().__init__()
        self.p = P
        self.k = K
        self.softmax = nn.Softmax(dim=0)
        self.gamma = self.init_gamma()
        self.beta_mu = nn.Parameter(torch.zeros((self.p,self.k)))
        #self.gamma = nn.Parameter(torch.rand(self.p, self.k))
        #self.beta_mu = nn.Parameter(torch.rand(self.p, self.k))
        self.beta_prior_tau = torch.tile(torch.tensor(1.0 / var_b * np.array([k+1 for k in range(self.k)]), dtype=torch.float32),(self.p,1))
        self.y_tau = torch.tensor(1.0 / (var_Y * (1-h2)), dtype=torch.float32)
        self.prior_pi = torch.ones((self.p,)) * (1/self.p)
        self.beta_post_tau = torch.tile(XX.reshape(-1,1),(1,self.k)) * self.y_tau + self.beta_prior_tau
        
    def init_gamma(self):
        weights = torch.tensor(1/self.p).repeat(self.p)
        multinomial = torch.multinomial(weights, self.k, replacement=True)
        one_hot = F.one_hot(multinomial, num_classes=self.p)
        return nn.Parameter(one_hot.type(torch.float32).T)

    '''
    def forward(self,XX,ytX,XtX,LD):
        # perform variational updates

        new_beta_mu = self.beta_mu.clone()
        new_gamma = self.gamma.clone()

        for k in range(self.k): # vectorize this code
            idxall = [x for x in range(self.k)]
            idxall.remove(k)
            beta_all_k = (self.gamma[:,idxall] * self.beta_mu[:,idxall]).sum(axis=1)
            new_beta_mu[:,k] = (ytX-torch.matmul(beta_all_k, XtX))/self.beta_post_tau[:,k] * self.y_tau
            u = -0.5*torch.log(self.beta_post_tau[:,k]) + torch.log(self.prior_pi.t()) + 0.5 * self.beta_mu[:,k]**2 * self.beta_post_tau[:,k]
            new_gamma[:,k] = self.softmax(u)
            #maxid = torch.argmax(u)
            #self.gamma[abs(LD[maxid])<0.05,k]= 0.0

        return new_beta_mu, new_gamma
    '''

    def forward(self, XX, ytX, XtX, LD): # LD not used in this function
        beta_all = (self.gamma * self.beta_mu).sum(axis=1)
        ll1 = self.y_tau * torch.matmul(beta_all,ytX)
        ll2 = - 0.5 * self.y_tau * ((((self.gamma * self.beta_mu**2).sum(axis=1) * XX).sum()))
        W = self.gamma * self.beta_mu
        WtRW = torch.matmul(torch.matmul(W.t(),XtX),W)
        ll3 = - 0.5 * self.y_tau * ( WtRW.sum() - torch.diag(WtRW).sum())
        ll = ll1 + ll2 + ll3
        betaterm1 = -0.5 * (self.beta_prior_tau * self.gamma * (self.beta_mu**2)).sum()
        gammaterm1 = (self.gamma * torch.tile(self.prior_pi.reshape(-1,1),(1,self.k))).sum() # confirm torch.tile = np.tile
        gammaterm2 = (self.gamma[self.gamma!=0] * torch.log(self.gamma[self.gamma!=0])).sum()
        mkl = betaterm1 + gammaterm1 - gammaterm2
        elbo = ll + mkl

        #t1 = self.gamma[self.gamma!=0]
        #t2 = torch.log(self.gamma[self.gamma!=0])
        #t3 = t1 * t2
        #t4 = t3.sum()

        #print('T1: ', torch.all(torch.isnan(t1)))
        #print('T2: ', torch.all(torch.isnan(t2)))
        #print('T3: ', torch.all(torch.isnan(t3)))
        #print('T4: ', torch.all(torch.isnan(t4)))
        return elbo
       
    def get_PIP(self):
        
        (val, idx) = torch.max((self.gamma),axis=1) # torch max returns a tuple
        return val
        
    # this func is not used
    def update_pi(self, new_pi):
        
        self.prior_pi = new_pi

    def get_effect_dict(self):
        
        numidx = (self.gamma>0.1).sum(axis=0)
        matidx = torch.argsort(-self.gamma, axis=0)

        result = dict()
        for i in range(self.k):
            if numidx[i] > 0:
                result[i] = matidx[0: numidx[i], i].tolist()

        #return {i:matidx[0:numidx[i],i].tolist() for i in range(self.k) if numidx[i]>0}
        return result

    def get_effect_num_dict(self):

        gamma = torch.round(self.gamma, decimals=4)
        beta_mu = torch.round(self.beta_mu, decimals=4)
        effect = self.get_effect_dict()
        eff_gamma = {i: np.round(gamma[effect[i], i].tolist(), 4) for i in effect}
        eff_mu = {i: np.round(beta_mu[effect[i], i].tolist(), 4) for i in effect}
        
        return eff_gamma, eff_mu

parser = argparse.ArgumentParser(description='SparsePro- Commands:')
parser.add_argument('--ss', type=str, default=None, help='path to summary stats', required=True)
parser.add_argument('--var_Y', type=float, default=None, help='GWAS trait variance', required=True)
parser.add_argument('--N', type=int, default=None, help='GWAS sample size', required=True)
parser.add_argument('--K', type=int, default=None, help='largest number of effect', required=True)
parser.add_argument('--LDdir', type=str, default=None, help='path to LD files', required=True)
parser.add_argument('--LDlst', type=str, default=None, help='path to LD list', required=True)
parser.add_argument('--save', type=str, default=None, help='path to save result', required=True)
parser.add_argument('--prefix', type=str, default=None, help='prefix for result files', required=True)
parser.add_argument("--verbose", action="store_true", help='options for displaying more information')
parser.add_argument("--tmp", action="store_true", help='options for saving intermediate file')
parser.add_argument("--ukb", action="store_true", help='options for using precomputed UK Biobank ld files from PolyFun')

args = parser.parse_args()

title()

if not os.path.exists(args.save):
    os.makedirs(args.save)

ss = pd.read_csv(args.ss,sep="\s+",dtype={'SNP':str,'BETA':float,'SE':float},index_col=0)
print("summary statistics loaded at {}".format(time.strftime("%Y-%m-%d %H:%M")))

ldlists=pd.read_csv(args.LDlst,sep='\s+',dtype={'ld':str,'start':int,'end':int})
print("LD list with {} LD blocks loaded\n".format(len(ldlists)))

pip = []
pip_name = []
cs = []
cs_pip = []
cs_eff = []
tl = []

for i in range(len(ldlists)):
    ld = ldlists['ld'][i]
    start = ldlists['start'][i]
    end = ldlists['end'][i]
    
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
    h2_hess,var_b=get_HESS_h2_SS(XtX,XX,LD.values,beta,se,args.N,args.var_Y)
    
    print("{} variants loaded from {} with {} variants having matched summary statistics explaining {:2.2%} of trait heritability \n".format(LD.shape[1], ld, len(idx), h2_hess))
    
    effidx = [i for i in range(len(idx)) if ((pos[i] >= start) & (pos[i] < end))]
    effnum = len(effidx)
    
    print('{} variants in the range of {} to {}'.format(effnum, start, end))
    if effnum <=20:
        print('Not enough effective variants, skipping')
        continue

    # note make_tensors() accepts LD.values (attribute) and returns LD_values (variable)
    XX, ytX, XtX, LD_values = make_tensors(XX, ytX, XtX, LD.values)
    model = SparsePro(len(beta),args.K,XX,args.var_Y,h2_hess,var_b) 
    opt = optim.Adam(model.parameters(), lr=1e-9, maximize=True, )

    # training loop
    for i in range(20):
        opt.zero_grad()
        loss = model(XX, ytX, XtX, LD) # use ELBO as loss function
        loss.backward()
        opt.step()

        if i % 5 == 0: print(loss.item())
        
    if args.tmp:
        #ll,mkl,elbo = model.loss(pred)
        loss = model(XX, ytX, XtX, LD)
        savelist = [h2_hess,var_b,model,loss]
        open_file = open(os.path.join(args.save,'{}.obj'.format(ld)),'wb')
        pickle.dump(savelist,open_file)
        open_file.close()
    
    mcs = model.get_effect_dict()
    eff_gamma, eff_mu = model.get_effect_num_dict()
    
    pip_tensor = model.get_PIP()
    pip_vec = torch.round(pip_tensor, decimals=4)
    pip.extend([pip_vec[i].item() for i in effidx])
    pip_name.extend([idx[i] for i in effidx])
    
    if len(mcs)==0:
        print("No effect detected")
        print()
        continue

    print("Detected k = {}".format(list(mcs)[-1]+1))
    print()
    for i in mcs:
        if mcs[i][0] in effidx:
            tl.append(idx[mcs[i][0]])
            mcs_idx = [idx[j] for j in mcs[i]]
            print('The {}-th effect contains effective variants:'.format(i))
            #print('causal variants: {}'.format(mcs_idx))
            print('casual variants: too many to print rn')
            print('posterior inclusion probabilities: {}'.format(eff_gamma[i]))
            print('posterior causal effect size: {}'.format(eff_mu[i])) 
            print()
            cs.append(mcs_idx)
            cs_pip.append(eff_gamma[i])
            cs_eff.append(eff_mu[i])


allPIP = pd.DataFrame({"idx":pip_name,"pip":pip})  
allPIP.to_csv(os.path.join(args.save,"{}.pip".format(args.prefix)),sep='\t',header=False,index=False)
allcs = pd.DataFrame({"cs":cs,"pip":cs_pip,"beta":cs_eff})
allcs.to_csv(os.path.join(args.save,"{}.cs".format(args.prefix)),sep='\t',header=True,index=False)
pd.DataFrame(tl,dtype='str').to_csv(os.path.join(args.save,"{}.tl".format(args.prefix)),sep='\t',header=False,index=False)
print("Statistical fine-mapping finished at {}. Writing all PIPs to {}.pip; all credible sets to {}.cs; all top snps in each effect to {}.tl ...".format(time.strftime("%Y-%m-%d %H:%M"),args.prefix,args.prefix,args.prefix))
