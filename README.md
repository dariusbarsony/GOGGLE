## GOGGLE: Generative Modelling of Tabular Data By Learning Relational Structure

PyTorch implementation of ICLR'23 paper [GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure](https://openreview.net/forum?id=fPVRcJqspu&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions)). Authors: Tennison Liu, Zhaozhi Qian, Jeroen Berrevoets, Mihaela van der Schaar

---
### Abstract

Generative modelling of tabular data entails a particular set of challenges, including heterogeneous relationships, limited number of samples, and difficulties in incorporating prior knowledge. This work introduces **GOGGLE**, a generative model that learns a relational structure underlying tabular data to better model variable dependencies, to introduce regularization, and to incorporate prior knowledge.

![GOGGLE Overview](./figures/goggle_recipe.png?raw=True)
**Key components of GOGGLE Framework.** 1. Simultaneous learning of relational structure $G_\phi$ and $F_\theta$ s.t. generative process respects relational structure. 2. Injection of prior knowledge and regularization on variable dependence. 3. Synthetic sample generated using $\hat{x} = F_\theta(z; G_\phi) \:, \: z\sim p_Z$.

---
### Experiments

To setup the virtual environment and necessary packages, please run the following commands:
```
$ conda create --name goggle_env python=3.8
$ conda activate goggle_env
```
Clone this repository and navigate to the root directory:
```
$ git clone https://github.com/tennisonliu/goggle.git
$ cd goggle
```
Install the required modules:
```
$ pip install -r requirements.txt
```

Place dataset in ```exps/data```, and see experiment notebooks with instructions in:
 - 1) ```exps/synthetic_data.``` for synthetic data generation
 - 2) ```exps/prior_knowledge/.``` for incorporating prior knowledge, and
 - 3) ```exps/ablation/.``` for ablation settings.

Data used in this work can be dowloaded using the following links:  

- 1) Adult: https://archive.ics.uci.edu/dataset/2/adult
- 2) Breast: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- 3) Covertype: https://archive.ics.uci.edu/dataset/31/covertype
- 4) Credit: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Additional datasets: 
- 5) ECOLI: 
- 6) MAGIC-IRRI: 
- 7) Red: https://archive.ics.uci.edu/dataset/186/wine+quality
- 8) White: (same as red)
- 9) Mice: https://archive.ics.uci.edu/dataset/342/mice+protein+expression
- 10) Musk: https://datahub.io/machine-learning/musk (UCI dowload not present)


The zip files (for Adult, Breast, Covertype, Credit, Red, White, Mice) should be unzipped in a folder named data in ```exps/```. Musk is a file and should be placed in ```exps/``` too. 

---

### Citation
If our paper or code helped you in your own research, please cite our work as:
```
@inproceedings{liu2023goggle,
  title={GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure},
  author={Liu, Tennison and Qian, Zhaozhi and Berrevoets, Jeroen and van der Schaar, Mihaela},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
