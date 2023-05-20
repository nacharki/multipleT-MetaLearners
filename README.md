# Meta-learners for estimating multi-valued treatments heterogeneuous effects.
Code Author: Naoufal Acharki (naoufal.acharki@gmail.com)

This repositery _multipleT-MetaLearners_ contains the code/simulations in R (see Appendices A-D) and the semi-synthetic dataset (described in Appendix E) as detailed in the Supplementary Materials of our the paper [Comparison of meta-learners for estimating multi-valued treatment heterogeneous effects (Acharki et al., 2023)](https://arxiv.org/abs/2205.14714)

Here, we implement different versions and meta-learners

### Experiments and simulations
All experiments in Acharki et al. (2023) can be replicated using this repository; the necessary code is in experiments.experiments_AISTATS21. To do so from shell, clone the repo, create a new virtual environment and run

Methods and scripts for the RPIE method are found in the fold R_code, and the analytical tests presented in the paper are can be found in test_functions



Note that PropensityMatching's methods require the following librairies: _numpy_, _pandas_, _seaborn_, _sklearn_ and _TableOne_.

### Semi-synthetic datasets:
In _Datasets_, you can find and upload the following datasets in a zip file:
- "Single_Fracture_Simulation_Cases_16200.csv"
- "Fracture_Efficency.csv"
- "Main_Dataset.csv"

We refer the reader to Appendix E for more details about the physical model used to generate this dataset and how it can be useful for further use/application in Causal Inference. An example of the use of this semi-synthetic dataset, the creation of a non-randomized biased dataset, is described in Appendix E and can be found in _Scripts/Appendix_EGS/EGS_CATE.

### Citation
If you use this software or the datasets please cite the corresponding paper(s):
```
@misc{acharki2023comparison,
      title={Comparison of meta-learners for estimating multi-valued treatment heterogeneous effects}, 
      author={Naoufal Acharki and Josselin Garnier and Antoine Bertoncello and Ramiro Lugo},
      year={2023},
      eprint={2205.14714},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
