Work of Gilead Turok and Ekene Ezeunala under [Prof David Knowles](https://davidaknowles.github.io).

# The Fine-Mapping Problem

Our work tries to solve the fine-mapping problem: determing which genotypes cause an observed phenotype.

Specifically, we are extending the SparsePro model ([preprint paper](https://www.biorxiv.org/content/10.1101/2021.10.04.463133v1) and [Github code](https://github.com/zhwm/SparsePro)) to better incorporate functional annotations. We use their code as a starting point, refactored it, and are now running our own experiments.

# Installing and Running

To run, first install all the requirments in the ```envs.yaml``` and ```requirments.txt``` files. Then nagivate to the directory ```sp``` and run

``` python src/main.py --opt cavi ```

To see all the possible flags, add a ```-h``` flag