## Introduction

Nowadays, millions of stray animals are alone on the streets or living in shelters, waiting for a new home. With the aim of better understanding the impact of pet photos and their features, [*PetFinder.my*](https://www.petfinder.my/) - Malaysia's leading animal welfare platform - designed a metric which measures pet photo popularity - *Pawpularity* - and hosted a related [Kaggle competition](https://www.kaggle.com/c/petfinder-pawpularity-score/overview).

This is our machine learning project where we build a predictive machine learning model that will be able to efficiently and accurately grade the pawpularity score, indicating how appealing the photograph of a pet offered for adoption on online platforms is.

## Methodology

### Cross-Validation
We use 5-fold Cross-Validation to train and validate models. The dataset is divided into 5 subsets, using a unique fold as a validation set and the remaining 4 as
the training set.  Using these subsets, we train 5 fold-models of the same architecture. 
![Cross-validation](figs/cross_validation.png)

### Forward Selection

Forward selection is a type of feature selection technique used primarily in statistical modeling and machine learning. The common use case is to select a subset of relevant features 
for model construction, improving model performance by reducing overfitting and enhancing generalization.  Using forward selection, we shortlisted 10 models of the pre-trained transformer models available in the Timm library that give the best results.
From an initial pool of 10 shortlisted models, the forward selection process was applied iteratively to identify the best combination of models that optimized performance.
The best-performing combination was achieved with 5 models. Adding additional models beyond this point not only failed to improve performance but also led to a slight decline. 
![Forward-selection](figs/forward_selection.png)
![Forward-selection](figs/num_models.png)

### Emsemble Learning

To further enhance the model’s performance, we used ensemble learning and weighted voting. Predictions from all trained fold models given certain weights to determine the final output. This method introduces diversity into the final
prediction process as it leverages the strengths of different model architectures.

Through trial and error, we achieved the best score using:
(0. 5 * 𝐵𝑒𝑖𝑡) + (0. 15 * 𝑉𝑖𝑡) + (0. 35 * 𝑆𝑤𝑖𝑛)


Further details on our solution can be found in the [report](https://github.com/sleepreap/Kaggle-Pawpularity/blob/main/Group02_report.pdf).

## Model used

- [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) 
- [Beit](https://arxiv.org/abs/2106.08254)
- [Vit](https://arxiv.org/abs/2010.11929)

# Results 
We can only compare with the leaderboard score as the competition is already over. We managed to get between 186th and 187th place, which is top 5.29%

![Results](figs/results.png)

## Authors
[Royce Teng](https://github.com/sleepreap) & [Yeek Sheng](https://github.com/Yeek020407) & [Yeek Sheng](https://github.com/Yeek020407)
