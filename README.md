<h1>DeepGS</h1>

Source Code Repository for DeepGS: Deep Representation Learning of Graphs and Sequences for Drug-Target Binding Affinity Prediction. Please refer to [our paper](http://ecai2020.eu/papers/34_paper.pdf) for detailed ([ECAI 2020](http://ecai2020.eu/) will be held soon )
<img src="figure1.png" alt="The framework of DeepGS" />

<h1>Installation</h1>

```bash
git clone https://github.com/jacklin18/DeepGS.git  
cd DeepGS  
pip install -r requirements.txt
```

<h1>Requirements</h1>


* [Pytorch](https://pytorch.org/)
* [RDKit](http://www.rdkit.org/docs/Install.html#how-to-install-rdkit-with-conda)
* scikit-learn
* Keras


<h1>Usage</h1>
(i) preprocess data as input

```bash
cd code
sh/bash preprocess.sh
```

(ii) train the model

```bash
sh/bash run_tranining.sh
```

<h1>Citation</h1>

If you use the code of DeepGS, please cite the [paper](http://ecai2020.eu/papers/34_paper.pdf) below:

> @article{lin2020deepgs,  
      title   ={DeepGS: Deep Representation Learning of Graphs and Sequences for Drug-Target Binding Affinity Prediction},  
      author  ={Lin, Xuan and Zhao, Kaiqi and Xiao, Tong and Quan, Zhe and Wang, Zhi-Jie and Yu, Philip S},  
      conference ={24th European Conference on Artificial Intelligence (ECAI)},  
      year    ={2020}  
  }
