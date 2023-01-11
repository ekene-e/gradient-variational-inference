Work of Gilead Turok and Ekene Ezeunala under [Prof David Knowles](https://davidaknowles.github.io).

# The Fine-Mapping Problem

Our work tries to **solve the fine-mapping problem**: determing which genotypes cause an observed phenotype. For example, identifying which Single Nucleotide Polymorphism (SNP) causes a particular disease.

We are **extending the SparsePro model** ([preprint paper](https://www.biorxiv.org/content/10.1101/2021.10.04.463133v1) and [Github code](https://github.com/zhwm/SparsePro)) to better incorporate functional annotations, biological and chemical information that allow us to know more about our genotypes. This additional information helps us better solve the fine-mapping problem.

More specifically, we use PyTorch's auto-differentiation capibilities to **smartly learn the functional annotation weight vector $w$ that determines which functional annotations are important**. We use the SparsePro code as a starting point, refactored it, and are now running our own experiments.

# Installing and Running

To run, first install all the requirments in the ```envs.yaml``` and ```requirments.txt``` files. Then nagivate to the directory ```sp``` and run

``` python src/main.py --variational-opt [cavi|adam] --weight-opt [adam|binary] ```

To see all the possible flags, add a ```-h``` flag
