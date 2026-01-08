# Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint

<!-- **Authors:** [Jiafei Lyu](https://dmksjfl.github.io/), Mengbei Yan, [Zhongjian Qiao](https://scholar.google.com/citations?user=rFU2fJQAAAAJ&hl=en&oi=ao), [Runze Liu](https://ryanliu112.github.io/), [Xiaoteng Ma](https://xtma.github.io/), [Deheng Ye](https://scholar.google.com/citations?user=jz5XKuQAAAAJ&hl=en&oi=ao), Jingwen Yang, [Zongqing Lu](https://z0ngqing.github.io/), [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en) -->

This is the codes for our method, Unbalanced Optimal Transport Data Filtering (UOTDF)

## Method Overview

<img src="https://github.com/dmksjfl/OTDF/blob/master/otdf.png" alt="image" width="600">

## How to run

Our project is based on [OTDF](https://github.com/dmksjfl/OTDF), therefore we share the same dependencies,  including `pytorch==1.8`, `gym==0.17.3` and `jax==0.4.9, jaxlib==0.4.9, ott-jax==0.4.5, jaxopt==0.8.3`, which are primarily used to calculate the optimal transportation scheme.

To reproduce our reported results in the submission, please check the following instructions:

**Step 1: Solve OT**

One has to first solve the unbalanced optimal transport problem with `run_ot.py` for every possible combination of source domain dataset and target domain dataset. We've included an example in the `compute_optimal_transport.sh` file, where the default key parameters include env, srctype, tartype, epsilon, lambda_src, lambda_tar, filter_threshold and metric. In this example, we use the default hyperparameter including `epsilon==0.01`, `lambda_src==0.05`, `lambda_tar==0.5`, and `filter_threshold==1.0`. 

This would produce a `hdf5` file in the `costlogs` directory. Note that this is mandatory before running UOTDF since it relies on the derived deviations for data filtering and weighting, otherwise an error would occur. In addition, we set a requirement to recalculate the optimal transport scheme for each seed before the start of the experiment.

**Step 2: Run UOTDF**

After Step 1, we can run UOTDF by calling

```
CUDA_VISIBLE_DEVICES=0 python train_otdf.py --env halfcheetah-morph --policy OTDF --srctype medium --tartype medium --weight --reg_weight 0.5 --seed 1
```

## Key Flags

For OTDF, one can specify how many source domain data to keep by `--proportion`, specify whether to include the weights on the source domain data by `--weight` (no weight flag indicates that source domain data will be equally treated), specify the dataset quality of the source domain by `--srctype`, and the dataset quality of the target domain by `--tartype`. One can determine the policy coefficient by specifying `reg_weight`. An example of running OTDF can be found below, and please see more details in `run.sh`,

```
# Example of running OTDF on morph task
CUDA_VISIBLE_DEVICES=0 python train_otdf.py 
    # environment name, e.g., halfcheetah / halfcheetah-morph / halfcheetah-gravity
    --env halfcheetah-morph 
    # policy
    --policy OTDF 
    # source domain dataset quality
    --srctype medium 
    # target domain dataset quality
    --tartype medium 
    # whether to add weight on source domain data
    --weight 
    # how many source domain data to keep
    --proportion 0.8
    # policy coefficient
    --reg_weight 0.5 
    # seed
    --seed 1
```