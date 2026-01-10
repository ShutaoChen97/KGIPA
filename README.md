<img width="432" height="235" alt="image" src="https://github.com/user-attachments/assets/0590b89e-06b9-46f0-81b4-0c2615b4b44d" /># KGIPA

<div align="center">

</div>

Natural language processing provides a systematic route for studying the functions of biological sequences such as proteins. However, modeling residue-level interactions remains challenging due to their intrinsic complexity and environment dependence. In this context, we introduce **KGIPA, a deep learning framework based on pragmatic analysis for inferring peptide-protein interactions**. KGIPA employs graph neural networks and a bilinear network to represent intra- and extra-linguistic contexts, thereby integrating multimodal molecular features and capturing environment-dependent interaction patterns. It further adopts a knowledge-guided strategy to coordinate diverse types of non-covalent bonds, revealing the non-covalent binding mechanisms between peptides and proteins. The model consistently outperforms state-of-the-art methods across cross-validation, unseen evaluation, and multi-label assessments, and demonstrates robust performance in peptide-protein binding affinity prediction and virtual screening of peptide-based therapeutics. **This work opens up opportunities for utilizing deep learning to infer complex molecular interactions**.

**Given the complexity and potential instability of individual environment configurations, we strongly recommend using the upgraded online prediction server of KGIPA, available at **http://106.63.6.174:21380/kgipa/**.
The previous version at http://bliulab.net/KGIPA/ remains accessible as well.**

![Model](/imgs/Model.png)

**Fig. 1: The model architecture of KGIPA.** KGIPA is a neural network model designed to achieve biological sequence pragmatic analysis, and it can be mainly divided into two parts: intra-linguistic and extra-linguistic contextual representation.

# 1 Installation

## 1.1 Create conda environment

```
conda create -n kgipa python=3.10
conda activate kgipa
```

## 1.2 Requirements
We recommend installing the environment using the provided `environment.yaml` file to ensure compatibility:
```
conda env update -f environment.yaml --prune
```


If this approach fails or Conda is not available, you can manually install the main dependencies as listed below:
```
python  3.10
biopython 1.84
huggingface-hub 0.26.1
numpy 2.1.2
transformers 4.46.0
tokenizers 0.20.1
sentencepiece 0.2.0
torch 2.5.0+cpu
torchaudio 2.5.0+cpu
torchvision 0.20.0+cpu
torch-geometric 2.6.1
```

> **Note** If you have an available GPU, the accelerated KGIPA can be used to predict peptide-protein binary interactions and pair-specific binding residues. Change the URL below to reflect your version of the cuda toolkit (cu118 for cuda=11.6 and cuda 11.8, cu121 for cuda 12.1). However, do not provide a number greater than your installed cuda toolkit version!
> 
> ```
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
> ```
>
> For more information on other cuda versions, see the [pytorch installation documentation](https://pytorch.org/).

## 1.3 Tools
Feature extraction tools and databases on which KGIPA relies: 
```
SCRATCH-1D 1.2
IUPred2A \
ncbi-blast 2.13.0
ProtT5 \
trRosetta \
```

Databases and model:
```
nrdb90 [ncbi-blast Database](http://bliulab.net/KGIPA/static/download/nrdb90.tar.gz)
uniclust30_2018_08 [HHsuite sequence Database](http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08)
model_res2net_202108 [Pre-trained network models of trRosetta](https://yanglab.qd.sdu.edu.cn/trRosetta/download/)
```

**The default paths to all tools and databases are shown in `conf.py`. You can change the paths to the tools and databases as needed by configuring `conf.py`.**

`SCRATCH-1D`, `IUPred2A`, `ncbi-blast`, `ProtT5`, and `trRosetta` are recommended to be configured as the system envirenment path. Your can follow these steps to install them:

### 1.3.1 How to install SCRATCH-1D
Download (For linux, about 6.3GB. More information, please see **https://download.igb.uci.edu/**)
```
wget https://download.igb.uci.edu/SCRATCH-1D_1.2.tar.gz
tar -xvzf SCRATCH-1D_1.2.tar.gz
```

Install
```
cd SCRATCH-1D_1.2
perl install.pl
```

> **Note:** The 32 bit linux version of blast is provided by default in the 'pkg' sub-folder of the package but can, probably should, and in some cases has to be replaced by the 64 bit or Mac OS version of the blast software for improved performances and compatibility on such systems.


Finally, test the installation of SCRATCH-1D
```
cd <INSTALL_DIR>/doc
../bin/run_SCRATCH-1D_predictors.sh test.fasta test.out 4
```

> **Note:** If your computer has less than 4 cores, replace 4 by 1 in the command line above.

**In addition, users need to move the `SCRATCH.sh` script in `/tools/SCRATCH-1D_1.2/` to the `bin/` directory under the SCRATCH-1D_1.2 installation path for KGIPA to call.**

### 1.3.2 How to install IUPred2A
For download and installation of IUPred2A, please refer to **https://iupred2a.elte.hu/download_new**. It should be noted that this automation service is **only applicable to academic users.** For business users, please contact the original authors for authorization. 

After obtaining the IUPred2A software package, decompress it.
```
tar -xvzf iupred2a.tar.gz
```

Finally, test the installation of IUPred2A
```
cd <INSTALL_DIR>
python3 iupred2a.py P53_HUMAN.seq long
```

**In addition, users need to move the `iupred2a.sh` script in `/tools/iupred2a/` to the IUPred2A installation path for KGIPA to call. Please also make sure to modify the path to the Python script (`iupred2a.py`) within the `iupred2a.sh` file to reflect the absolute path of your IUPred2A installation. This is necessary for the script to function correctly.**

### 1.3.3 How to install ncbi-blast
Download (For x64-linux, about 220M. More information, please see **https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html**)
```
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.13.0/ncbi-blast-2.13.0+-x64-linux.tar.gz
tar -xvzf ncbi-blast-2.13.0+-x64-linux.tar.gz
```
Add the path to system envirenment in `~/.bashrc`.
```
export BLAST_HOME={your_path}/ncbi-blast-2.13.0+
export PATH=$PATH:$BLAST_HOME/bin
```

Finally, reload the system envirenment and check the ncbi-blast command:
```
source ~/.bashrc
psiblast -h
```

> **Note:** The purpose of KGIPA with the help of ncbi-blast is to extract the position-specific scoring matrix (PSSM). It should be noted that for sequences that cannot be effectively aligned, the PSSM is further extracted by blosum62 (which can be found in [`blosum62.txt`](http://bliulab.net/KGIPA/static/download/blosum62.txt)).


### 1.3.4 How to install ProtT5 (ProtT5-XL-UniRef50)
Download and install (More information, please see **https://github.com/agemagician/ProtTrans** or **https://zenodo.org/record/4644188**, about 5.3GB)

```
wget https://zenodo.org/records/4644188/files/prot_t5_xl_uniref50.zip?download=1
unzip prot_t5_xl_uniref50.zip
```


### 1.3.5 How to install trRosetta
Download `trRosettaX.tar.bz2` from **https://yanglab.qd.sdu.edu.cn/trRosetta/download/** (about 94MB)

```
tar -xvjf trRosettaX.tar.bz2
cd trRosettaX/
```

Then, refer to the README file to install and configure it. **In addition, users need to move the `generate_msa.sh` and `predict.sh` scripts in `/tools/trRosettaX/` to the trRosettaX installation path for KGIPA to call.**
Please also ensure the following modifications are made for proper execution:

**1) Conda Environment:**
Both `generate_msa.sh` and `predict.sh` need to run under the Conda environment required by trRosettaX.
This can be achieved by modifying the `export PATH=...` line in each script to point to the bin directory of the corresponding Conda environment.

**2) Database Path Configuration:**

In `generate_msa.sh`, set the `-hhdb` parameter to the absolute path of the uniclust30_2018_08 database.

In `predict.sh`, set the `-mdir` parameter to the absolute path of the model_res2net_202108 model directory.

These changes are necessary to ensure KGIPA can correctly invoke trRosetta for structure prediction.


## 1.4 Install KGIPA
To install from the development branch run
```
git clone git@github.com:ShutaoChen97/KGIPA.git
cd KGIPA/
```

**Finally, configure the defalut path of the above tool and the database in `conf.py`. You can change the path of the tool and database by configuring `conf.py` as needed.**


# 2 Usage
It takes 2 steps to predict peptide-protein binary interaction and peptide-protein-specific binding residues:

(1) Replace the default peptide sequence in the `example/Peptide_Seq.fasta` file with your peptide sequence (FASTA format). Similarly, replace the default protein sequence in the `example/Protein_Seq.fasta` file with your protein sequence (FASTA format). If you don't want to do this, you can also test your own peptide-protein pairs by modifying the paths to the files passed in by the `run_predictor.py` script (the parameter is `-uip`, respectively).

(2) Then, run `run_predictor.py` to make prediction, including **pairwise non-covalent interaction** prediction and **non-covalent bond type** identification. It should be noted that `run_predictor.py` automatically calls the scripts `FeatureExtract.py`, and `PSSMExtract.py` to generate the multi-source isomerization features of peptides and proteins.
```
conda activate kgipa
python run_predictor.py -uip example
```

If you want to retrain based on your private dataset, find the original KGIPA model in `model.py`. The KGIPA source code we wrote is based on the Pytorch implementation and can be easily imported by instantiating it.

# 3 Problem feedback
If you have questions on how to use KGIPA, feel free to raise questions in the [discussions section](https://github.com/ShutaoChen97/KGIPA/discussions). If you identify any potential bugs, feel free to raise them in the [issuetracker](https://github.com/ShutaoChen97/KGIPA/issues).

In addition, if you have any further questions about KGIPA, please feel free to contact us [**stchen@bliulab.net**]

# 4 Citation

If you find our work useful, please cite us at
```
@article{
  title={A deep learning framework based on pragmatic analysis for inferring peptide-protein non-covalent interactions},
  author={Shutao Chen, Ke Yan, Jiangyi Shao, Xiangxiang Zeng, and Bin Liu},
  journal={submitted},
  year={2026},
  publisher={}
}

```
