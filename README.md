<h1>DeepGS</h1>

Source Code Repository for DeepGS: Deep Representation Learning of Graphs and Sequences for Drug-Target Binding Affinity Prediction. Please refer to [our paper](https://arxiv.org/pdf/2003.13902.pdf) for detailed ([ECAI 2020](http://ecai2020.eu/) will be held soon )
<img src="figure1.png" alt="The framework of DeepGS" />

<h1>Installation</h1>

```bash
git clone https://github.com/jacklin18/DeepGS.git  
cd DeepGS  
pip install -r requirements.txt
```

<h1>Requirements</h1>

```bash
-Pytorch
-RDKit
-scikit-learn
-Keras<br>
```

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

If you use the code of DeepGS, please cite the [paper](https://arxiv.org/abs/2003.13902) below:

```bash
@article{Lin2020DeepGSDR,
  title={DeepGS: Deep Representation Learning of Graphs and Sequences for Drug-Target Binding Affinity Prediction},
  author={Xuan Lin and Kaiqi Zhao and Tong Xiao and Zhe Quan and Zhi-Jie Wang and Philip S. Yu},
  journal={ArXiv},
  year={2020},
  volume={abs/2003.13902}
}
```
