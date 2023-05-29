# Meta-learners for estimating multi-valued treatments heterogeneuous effects.
Code Author: Naoufal Acharki (naoufal.acharki@gmail.com)

This repositery _multipleT-MetaLearners_ contains the code/simulations in R (see Appendices A-D) and the semi-synthetic dataset (described in Appendix E) as detailed in the Supplementary Materials of our the paper [Comparison of meta-learners for estimating multi-valued treatment heterogeneous effects (Acharki et al., 2023)](https://arxiv.org/abs/2205.14714)

Here, we implement meta-learners (T-, S-, M-, DR-, R-) and different versions (X- and Naive X-) learners for multi-valued treatments to estimate heterogenous treatments. These implementations can be found in _MetaLearners_tools.R_ in _Scripts_ and the analytical tests presented in the paper (linear model and Hazard rate) are written in each scripts of Appendix_D1 to Appendix_D4.

This software is currently in beta, and we expect to make continual improvements to its performance and usability.

### Experiments and simulations
All experiments in [Acharki et al. (2023)](https://arxiv.org/abs/2205.14714) can be replicated using this repository in the fold _Scripts_. The necessary code for each table in the main paper or Supplementary Materials can be reproduced by running to script associated to the experiment (e.g. Lin_Rand_Case1.R in Appendix D1 for Table 5). 

### Semi-synthetic datasets:
In the fold _Datasets_, you can find and upload the following datasets in the zip file _Semi-Synthetic-EGS.zip_ :
- "Single_Fracture_Simulation_Cases_16200.csv"
- "Fracture_Efficency.csv"
- "Main_Dataset.csv"

We refer the reader to Appendix E for more details about the physical model used to generate this dataset and how it can be useful for further use/application in Causal Inference. An example of the use of this semi-synthetic dataset, the creation of a non-randomized biased dataset, is described in Appendix E and can be found in _Scripts/Appendix_EGS/EGS_CATE_.

### Citation
If you use this software or the datasets please cite the corresponding paper(s):
```
@misc{acharki2023comparison,
      title={Comparison of meta-learners for estimating multi-valued treatment heterogeneous effects}, 
      author={Naoufal Acharki and Ramiro Lugo and Antoine Bertoncello and Josselin Garnier},
      year={2023},
      eprint={2205.14714},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
