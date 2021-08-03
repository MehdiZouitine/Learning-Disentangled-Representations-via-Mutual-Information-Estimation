# Learning Disentangled Representations via Mutual Information Estimation

**Pytorch** Implementation of **Learning Disentangled Representations via Mutual Information Estimation** ([arxiv link](https://arxiv.org/abs/1912.03915)) by Eduardo Hugo Sanchez *et al.*

The implementation is done in pytorch on the **colored-mnist dataset**.
<p align="center">
<img  src="https://github.com/MehdiZouitine/spaghetti/blob/master/images/pair.png?raw=true" alt="Cmnist">
</p>



The training is divided into two stages : 
* First, the shared representation is learned via cross mutual information estimation and maximization.
* Secondly, mutual information maximization is performed to learn the exclusive representation while minimizing the mutual information between the shared and exclusive representations (using an adversarial objective).

<p align="center">
<img  src="https://github.com/MehdiZouitine/spaghetti/blob/master/images/disen.PNG?raw=true" alt="Pipeline">
</p>

## Installation
```
git clone https://github.com/MehdiZouitine/spaghetti.git
cd Learning-Disentangled-Representations-via-Mutual-Information-Estimation/
pip install -r requirement.txt
```

## Learn shared representation 

To run the first stage of the training, one may use **sdim_trainer.sh**

```
echo Start shared representation training
data_base_folder="data"
xp_name="Share_representation_training"
conf_path="conf/share_conf.yaml"
```
* data_base_folder : Is the folder where the raw mnist data is hosted. By default this folder is called "data" and the data is downloaded the first time this file is run.

* xp_name : Mlflow experimentation name.

* conf_path : Path to the training configuration file. To use sdim_trainer.sh the conf file must be shaped like **share_conf.yaml** .


## Learn exclusive representation 

To run the first stage of the training, one may use **sdim_trainer.sh** to get pretrained shared encoder and then use
**edim_trainer.sh**.

```
echo Start exclusive representation training
data_base_folder="data"
xp_name="Exclusive_representation_training"
conf_path="conf/exclusive_conf.yaml"
trained_enc_x_path="mlruns/3/38e65dbd8d1246fab33f079e16510019/artifacts/sh_encoder_x/state_dict.pth"
trained_enc_y_path="mlruns/3/38e65dbd8d1246fab33f079e16510019/artifacts/sh_encoder_y/state_dict.pth"
```
* data_base_folder : Is the folder where the raw mnist data is hosted. By default this folder is called "data" and the data is downloaded the first time this file is run.

* xp_name : Mlflow experimentation name.

* conf_path : Path to the training configuration file. To use edim_trainer.sh the conf file must be shaped like **exclusive_conf.yaml**.

* trained_enc_x_path : Path the the pretrained encoder of domains X. As you can see encoders are logged in mlflow.

* trained_enc_y_path : Path the the pretrained encoder of domains Y. As you can see encoders are logged in mlflow.

## Makefile

Once the *sdim_runner.sh* and *edim_runner.sh* file are completed, the user can launch a training session using the Makefile shortcut.
```
share_train:
	bash sdim_runner.sh

exclusive_train:
	bash edim_runner.sh

board:
	pkill gunicorn || true
	mlflow ui
```

* share_train : Train shared encoders using the **sdim_runner.sh** configuration.
* exclusive_train : Train the exclusive encoders using the **edim_runner.sh** configuration.
* board : Launch the mlflow monitoring windows on localhost.
