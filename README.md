<h1>DeepGS</h1>

Source Code Repository for DeepGS: Deep Representation Learning of Graphs and Sequences for Drug-Target Binding Affinity Prediction. Please refer to [our paper](https://arxiv.org/pdf/2003.13902.pdf) for detailed ([ECAI 2020](http://ecai2020.eu/) will be held soon )
<img src="figure1.png" alt="The framework of DeepGS" />

<h1>Installation</h1>

```bash
git clone https://github.com/jacklin18/DeepGS.git  
cd DeepGS  
python -r requirements.txt install
```

<h1>Requirements</h1>

```bash
-Pytorch<br>
-RDKit<br>
-scikit-learn<br>
-Keras<br>
```

<h1>Usage</h1>
(i) preprocess data as input<br>

```bash
cd code<br>
sh/bash preprocess.sh<br>
```

(ii) train the model<br>

```bash
sh/bash run_tranining.sh<br>
```
