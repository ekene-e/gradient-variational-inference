# SparsePro for efficient genome-wide fine-mapping with summary statistics and functional annotations

SparsePro is a command line tool for efficiently conducting genome-wide fine-mapping. Our method has two key features: First, by creating a sparse low-dimensional projection of the high-dimensional genotype, we enable a linear search of causal variants instead of an exponential search of causal configurations in most existing methods; Second, we adopt a probabilistic framework with a highly efficient variational expectation-maximization algorithm to integrate statistical associations and functional priors.

Full description is available in our [preprint paper](https://www.biorxiv.org/content/10.1101/2021.10.04.463133v1). 

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Input files](#input-files)
- [Usage](#usage)
  * [SparsePro-: statistical fine-mapping with summary statistics](#sparsepro---statistical-fine-mapping-with-summary-statistics)
  * [Enrich: testing function enrichment of annotations](#enrich--testing-function-enrichment-of-annotations)
  * [SparsePro+: annotated fine-mapping with summary statistics and functional annotations](#sparsepro---annotated-fine-mapping-with-summary-statistics-and-functional-annotations)
  * [Genome-wide fine-mapping with pre-computed UK Biobank LD matrix](#genome-wide-fine-mapping-with-pre-computed-uk-biobank-ld-matrix)
- [Output files](#output-files)
- [FAQ](#faq)
- [License](#license)
- [Authors](#authors)
- [Citations](#citations)

## Overview 

<img align="right" src="doc/Fig1.png" width=60% height=60%>
Identifying causal variants from genome-wide association studies (GWASs) is challenging due to widespread linkage disequilibrium (LD). Functional annotations of the genome may help prioritize variants that are biologically relevant and thus improve fine-mapping of GWAS results.

To fine-map causal SNPs, our method takes two lines of evidence. First, from estimated marginal associations between genetic variants and a complex trait of interest, accompanied by matched LD information, we can group correlated genetic variants together and assess their effects jointly. Then we infer the contribution of each SNP towards each group of causal effect separately to obtain posterior inclusion probabilities (PIPs). Second, optionally, if we have knowledge about any functional annotations which may be enriched for the causal SNPs, we can estimate the relative enrichment of these annotations, and prioritize SNPs with these annotations so that they are more likely to be considered causal variants. SparsePro yields functionally informed PIP for each SNP and the enrichment estimates of candidate functional annotations.

## Installation

SparsePro was developed under Python 3.9.7 environment but should be compatible with older versions of Python 3. The following Python modules are required:

* [numpy](http://www.numpy.org/) (version==1.21.3)
* [scipy](http://www.scipy.org/) (version==1.7.1)
* [pandas](https://pandas.pydata.org/getpandas.html) (version==1.3.4)

To install SparsePro:

```
git clone https://github.com/zhwm/SparsePro.git
cd SparsePro
pip install -r requirements.txt 
``` 

To test the installation and display basic usage:

```
$> python src/sparsepro.py -h
usage: sparsepro.py [-h] --ss SS --var_Y VAR_Y --N N --K K --LDdir LDDIR --LDlst LDLST --save SAVE --prefix PREFIX [--verbose] [--tmp] [--ukb]

SparsePro- Commands:

optional arguments:
  -h, --help       show this help message and exit
  --ss SS          path to summary stats
  --var_Y VAR_Y    GWAS trait variance
  --N N            GWAS sample size
  --K K            largest number of effect
  --LDdir LDDIR    path to LD files
  --LDlst LDLST    path to LD list
  --save SAVE      path to save result
  --prefix PREFIX  prefix for result files
  --verbose        options for displaying more information
  --tmp            options for saving intermediate file
  --ukb            options for using precomputed UK Biobank ld files from PolyFun
```


```
$> python src/enrich.py -h   
usage: enrich.py [-h] --save SAVE --prefix PREFIX --anno ANNO --pip PIP --pthres PTHRES

SparsePro Enrich Commands:

optional arguments:
  -h, --help       show this help message and exit
  --save SAVE      path to save result
  --prefix PREFIX  prefix for result files
  --anno ANNO      path to annotation file
  --pip PIP        path to pip file
  --pthres PTHRES  p value threshold for enrichment
```

```
$> python src/sparsepro_plus.py -h
usage: sparsepro_plus.py [-h] --ss SS --var_Y VAR_Y --N N --K K --LDdir LDDIR --LDlst LDLST --save SAVE --prefix PREFIX [--anno ANNO] --W W [--verbose]
                         [--ukb]

SparsePro+ Commands:

optional arguments:
  -h, --help       show this help message and exit
  --ss SS          path to summary stats
  --var_Y VAR_Y    GWAS trait variance
  --N N            GWAS sample size
  --K K            largest number of effect
  --LDdir LDDIR    path to LD files
  --LDlst LDLST    path to LD list
  --save SAVE      path to save result
  --prefix PREFIX  prefix for result files
  --anno ANNO      path to annotation file
  --W W            path to enriched file
  --verbose        options for displaying more information
  --ukb            options for using precomputed UK Biobank ld files from PolyFun
```

## Input files

Example input files are included in the [dat](dat/) directory.

SparsePro takes in the following files: 

1. **summmary statistics file** that contains SNP ('CHR.POS.A1.A2'; must match the IDs used in LD file(s)), BETA (effect size estimate from GWAS), SE (standard deviation of effect size from GWAS).

```
$> head -5 dat/22.ss

SNP	BETA	SE
22.19500559.G.A	0.00562955	0.0322818
22.19500581.C.A	0.00258343	0.00596305
22.19500657.C.G	0.00298233	0.00267837
22.19500832.T.C	0.0161714	0.0101509 
```

2. **LD file(s)** that contains SNP-SNP correlations. The indices should match the summary statistics file.   

```
$> head -5 dat/chr22_19500001_21000001.ld | cut -f 1-5
SNP	22.19500559.G.A	22.19500581.C.A	22.19500657.C.G	22.19500832.T.C
22.19500559.G.A	1.0	-0.0169	-0.063	-0.0102
22.19500581.C.A	-0.0169	1.0	-0.2274	-0.0317
22.19500657.C.G	-0.063	-0.2274	1.0	-0.1293
22.19500832.T.C	-0.0102	-0.0317	-0.1293	1.0
```

3. **list of LD file(s)** contains a list of LD file(s). The SNPs in these LD files should match SNPs in the summary statistics file. The start and end columns indicate the range in each LD matrix being used.

```
$> cat dat/22.lst 
ld	start	end
chr22_19500001_21000001.ld.gz	19500001	20500001
chr22_20000001_21500001.ld.gz	20500001	21000001
chr22_20500001_22000001.ld.gz	21000001	22000001
```
4. (optional) **annotation file** with binary entries indicating whether SNPs have the corresponding annotations. 

```
$> head -5 dat/22.anno | cut -f 1-5
SNP	Conserved_LindbladToh	DHS_Trynka	H3K27ac_Hnisz	H3K4me3_Trynka
22.19500559.G.A	0	0	1	0
22.19500581.C.A	0	0	1	0
22.19500657.C.G	0	0	1	0
22.19500832.T.C	0	1	1	0
```

## Usage

Here we use a part of FEV1/FVC ratio GWAS summary statistics calculated using UK Biobank European ancestry participants to showcase genome-wide fine-mapping procedures with SparsePro. All files are included in the [dat](dat/) directory.

### SparsePro-: statistical fine-mapping with summary statistics

We use [sparsepro.py](src/) to perform statistical fine-mapping. We can provide the summary statistic file, LD files directory, and LD list file through `--ss`, `--LDdir`, and `--LDlst`, respectively. 

To help setting hyperparameters, GWAS sample size `--N` is set as 283677 and trait variance `--var_Y` is provided as 1.0 since we performed inverse normal transformation before GWAS. We also set the largest number of effect `--K` to be 9.

Also, we use `--save` to specify the path for saving results and `--prefix` 22 to specify the prefix for result files.

If you intend to perform annotated fine-mapping, please use `--tmp`, to store intermediate files (with suffix .obj) to save computation time.

We suggest separating the whole genome into chromosomal chunks to perform parallel statistical fine-mapping. 

```python
python src/sparsepro.py \
    --ss dat/22.ss \
    --var_Y 1.0 \
    --N 283677 \
    --LDdir dat/LD \
    --LDlst dat/22.lst \
    --save dat/res \
    --prefix 22 \
    --tmp \
    --K 9
```

Here is the expected output:

```
summary statistics loaded at 2021-11-25 12:38
LD list with 3 LD blocks loaded

6134 variants loaded from chr22_19500001_21000001.ld.gz with 6134 variants having matched summary statistics explaining 0.21% of trait heritability 

4669 variants in the range of 19500001 to 20500001
Detected k = 4

The 1-th effect contains effective variants:
causal variants: ['22.19753449.A.G', '22.19754091.A.C', '22.19753848.A.G', '22.19750773.T.C']
posterior inclusion probabilities: [0.417, 0.1456, 0.1172, 0.1008]
posterior causal effect size: [0.0176, 0.0167, 0.0166, 0.0165]

The 3-th effect contains effective variants:
causal variants: ['22.19803382.C.A']
posterior inclusion probabilities: [0.218]
posterior causal effect size: [-0.0148]

5804 variants loaded from chr22_20000001_21500001.ld.gz with 5804 variants having matched summary statistics explaining 0.18% of trait heritability 

1465 variants in the range of 20500001 to 21000001
Detected k = 2

The 0-th effect contains effective variants:
causal variants: ['22.20778066.A.G', '22.20776406.A.G']
posterior inclusion probabilities: [0.472, 0.1045]
posterior causal effect size: [-0.0354, -0.0349]

The 1-th effect contains effective variants:
causal variants: ['22.20780296.G.A']
posterior inclusion probabilities: [0.9462]
posterior causal effect size: [0.0314]

4561 variants loaded from chr22_20500001_22000001.ld.gz with 4561 variants having matched summary statistics explaining 0.16% of trait heritability 

3096 variants in the range of 21000001 to 22000001
Detected k = 2

Statistical fine-mapping finished at 2021-11-25 12:39. Writing all PIPs to 22.pip; all credible sets to 22.cs; all top snps in each effect to 22.tl ...
```

### Enrich: testing function enrichment of annotations

After obtaining PIPs from statistical fine-mapping, we can test for functional enrichment of annotations with [enrich.py](src/).

We use `--anno` to provide the annotation file and `--pip` to provide statistical fine-mapping PIPs. To increase statistical power of functional annotations, we suggest aggregating whole-genome SNPs to perform the testing. 

We use the p-value cutoff supplied to `--pthres` to select significantly enriched annotations for updating priors in annotated fine-mapping. Here we use 0.3 only to showcase the utility. We recommend setting the threshold to **1e-6** if genome-wide SNPs were included.

```python
python src/enrich.py \
    --save dat/res \
    --prefix 22 \
    --anno dat/22.anno \
    --pip dat/res/22.pip \
    --pthres 0.3
```

Here is the expected output:

```
Annotation file Loaded at 2021-11-25 12:45
There are 9230 variants with 10 annotations and among them 9230 variants have PIP esitmates

Univariate testing finished at 2021-11-25 12:45. Saving result to 22.wsep file...

4 annotations are deemed significantly enriched at 0.3 p-value threshold and used to update priors. Saving result to 22.W0.3 file...
```

### SparsePro+: annotated fine-mapping with summary statistics and functional annotations 

We use [sparsepro_plus.py](src/) to perform annotated fine-mapping. Given the relative enrichment estimates, we can update the prior probability of being causal for each variants with their annotations.

Please use the same `--K`, `--N` and `--var_Y` for both statistical fine-mapping and annotated fine-mapping. 

```python
python src/sparsepro_plus.py \
    --ss dat/22.ss \
    --var_Y 1.0 \
    --N 283677 \
    --LDdir dat/LD \
    --LDlst dat/22.lst \
    --save dat/res \
    --prefix 22 \
    --K 9 \
    --anno dat/22.anno \
    --W dat/res/22.W0.3 
```

Here is the expected output:

```
summary statistics loaded at 2021-11-25 12:46
Annotation file Loaded at 2021-11-25 12:46
There are 9230 variants with 10 annotations and among them 9230 variants have summary statistics

LD list with 3 LD blocks loaded

6134 variants loaded from chr22_19500001_21000001.ld.gz with 6134 variants having matched summary statistics explaining 0.21% of trait heritability 

4669 variants in the range of 19500001 to 20500001
Detected k = 4

The 1-th effect contains effective variants:
causal variants: ['22.19754091.A.C']
posterior inclusion probabilities: [0.7886]
posterior causal effect size: [0.0166]

The 3-th effect contains effective variants:
causal variants: ['22.19803382.C.A', '22.19798836.C.G']
posterior inclusion probabilities: [0.1356, 0.1037]
posterior causal effect size: [-0.0147, -0.0136]

5804 variants loaded from chr22_20000001_21500001.ld.gz with 5804 variants having matched summary statistics explaining 0.18% of trait heritability 

1465 variants in the range of 20500001 to 21000001
Detected k = 2

The 0-th effect contains effective variants:
causal variants: ['22.20785639.G.A', '22.20778066.A.G']
posterior inclusion probabilities: [0.7394, 0.1459]
posterior causal effect size: [0.035, -0.0356]

The 1-th effect contains effective variants:
causal variants: ['22.20780296.G.A']
posterior inclusion probabilities: [0.9971]
posterior causal effect size: [0.0308]

4561 variants loaded from chr22_20500001_22000001.ld.gz with 4561 variants having matched summary statistics explaining 0.16% of trait heritability 

3096 variants in the range of 21000001 to 22000001
Detected k = 2

Annotated fine-mapping finished at 2021-11-25 12:47. Writing all PIPs to 22.apip; all credible sets to 22.acs; all top snps in each effect to 22.atl ...
```

So far, we have finished fine-mapping our subset of the FEV1/FVC ratio GWAS summary statistics. Due to the small number of variants investigated, we do not find any evidence of functional enrichment. We can visualize the statistical fine-mapping PIPs and compare with the p-values obtained in GWAS.

<img src='doc/showcase.png'>


### Genome-wide fine-mapping with pre-computed UK Biobank LD matrix

Since the computation of LD can take quite some time, we also provide an option `--ukb` to use pre-computed UK Biobank whole-genome LD matrix provided in [PolyFun](https://www.nature.com/articles/s41588-020-00735-5). They can be downloaded from [here](https://alkesgroup.broadinstitute.org/UKBB_LD/).

In order to match the summary statistics with these LD matrix, please make sure the SNP index are matched with those provided in [idx](ukb/idx/) directory. This can be achieved by flipping the sign of BETA in the summary statistics to match with the index. Using the `--ukb` flag with the directory of the downloaded LD file provided by `--LDdir`, we can perform SparsePro-, Enrich and SparsePro+ as described before. Here are some examples:

```
python src/sparsepro.py --ukb --ss FFR_22.ss --var_Y 1.0 --N 283677 --K 9 --LDdir ~/UKBBLD/ --LDlst ukb/lst/22.lst --save res --prefix FFR_22 --tmp
```

```
python src/enrich.py --save res --prefix FFR_22 --anno ukb/anno/UKBB.22.anno --pip res/FFR_22.pip --pthres 1e-3
```

```
python src/sparsepro_plus.py --ukb --ss FFR_22.ss --var_Y 1.0 --N 283677 --K 9 --LDdir ~/UKBBLD/ --LDlst ukb/lst/22.lst --save res --prefix FFR_22 --anno ukb/anno/UKBB.22.anno --W res/FFR_22.W1e-3
```

## Output files

If no functional annotation is provided, we have the following output files saved to the path specified by --save:

1. statistical fine-mapping PIPs

```
$> head -5 22.pip
22.19500559.G.A	0.0002
22.19500581.C.A	0.0001
22.19500657.C.G	0.0001
22.19500832.T.C	0.0002
22.19500918.T.C	0.0002
```

2. statistical fine-mapping effects (credible sets): each row represents an effect; the cs column contains the SNPs included in the effect; the pip column contains the SNPs with PIP greater than 0.1 in the effect; and the beta column contains effect sizes of these SNPs if selected to be causal in this effect. 

```
$> cat 22.cs
cs	pip	beta
['22.19753449.A.G', '22.19754091.A.C', '22.19753848.A.G', '22.19750773.T.C']	[0.417, 0.1456, 0.1172, 0.1008]	[0.0176, 0.0167, 0.0166, 0.0165]
['22.19803382.C.A']	[0.218]	[-0.0148]
['22.20778066.A.G', '22.20776406.A.G']	[0.472, 0.1045]	[-0.0354, -0.0349]
['22.20780296.G.A']	[0.9462]	[0.0314]
```

3. statistical fine-mapping top list: a list of the most representative SNP in each effect
```
$> cat 22.tl
22.19753449.A.G
22.19803382.C.A
22.20778066.A.G
22.20780296.G.A
```

Given functional annotations, we have the following additional output files saved to the path specified by --save:

1. annotated fine-mapping PIPs

```
$> head -5 22.apip
22.19500559.G.A	0.0001
22.19500581.C.A	0.0
22.19500657.C.G	0.0
22.19500832.T.C	0.0001
22.19500918.T.C	0.0001
```

2. statistical fine-mapping effects (credible sets)

```
$> cat 22.acs
cs	pip	beta
['22.19754091.A.C']	[0.7886]	[0.0166]
['22.19803382.C.A', '22.19798836.C.G']	[0.1356, 0.1037]	[-0.0147, -0.0136]
['22.20785639.G.A', '22.20778066.A.G']	[0.7394, 0.1459]	[0.035, -0.0356]
['22.20780296.G.A']	[0.9971]	[0.0308]
```

3. statistical fine-mapping top list
```
$> cat 22.atl
22.19754091.A.C
22.19803382.C.A
22.20785639.G.A
22.20780296.G.A
```

4. univariate test results of functional enrichment for annotations. The first column is the relative enrichment estimates; the second column is standard deviations of relative enrichment estimates; and the third column is the p-value of annotation enrichment.

```
$> head -5 22.wsep
	W	se	p
Conserved_LindbladToh	2.2125	0.9555	0.2115
DHS_Trynka	1.2337	0.8341	0.3256
H3K27ac_Hnisz	1.1693	1.195	0.5363
H3K4me3_Trynka	1.2785	0.8507	0.2731
```

5. joint relative enrichment estimation of significantly enriched annotations (at the p-value cutoff supplied to --pthres). The first column contains significantly enriched annotations; the second and the third columns contain joint estimates of relative enrichement and their standard deviations. The fourth column contains the column index of these annotations in the original annotation file (0-based).

```
>$ head 22.W0.3 
ANNO	W_sig	W_se_sig	sigidx
Conserved_LindbladToh	1.1045804382746935	0.9564783649206201	0
H3K4me3_Trynka	1.0370385745188193	0.8507116428093798	3
non_synonymous	2.2952201065704325	1.0265360522406743	8
Human_Promoter_Villar_ExAC	-1.2941718438595082	7.100477456141411	9
```



## FAQ

1. How do we obtain trait variance for --var_Y from summary statistics?

   - If the trait has been standardized to have unit variance prior to performing GWAS (for example, inverse normal transformed), we can set it as 1.0. Otherwise, it can be estimated from `var_Y = 2Np(1-p)se^2`  where `N` (the sample size), `p` (minor allele frequencies), and `se` (standard errors of effect size estimates) are usually available in GWAS summary statistics. If the `var_Y` estimates vary across variants, we supply the median value of all these estimates.

2. How do we set hyperparameter K?

   - We have shown that SparsePro is not sensitive to the setting of K as long as K is larger than the actual number of causal effects, except that increasing K marginally increases the computation time.   

## License

This project is licensed under the MIT License.

## Authors

- Wenmin Zhang (wenmin.zhang@mail.mcgill.ca)
- Hamed Najafabadi (hamed.najafabadi@mcgill.ca)
- Yue Li (yueli@cs.mcgill.ca)

## Citations

If you use this software, please cite:

[Wenmin Zhang, Hamed Najafabadi, Yue Li. SparsePro: an efficient genome-wide fine-mapping method integrating summary statistics and functional annotations. bioRxiv 2021.10.04.463133](https://doi.org/10.1101/2021.10.04.463133)
