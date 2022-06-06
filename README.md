# ARBRE_pytorch

The source code of ARBRE (Asymmetrical Context-aware Modulation for Collaborative Filtering Recommendation).

## Environment Requirements

python: 3.7.11
pytorch: 1.7.1
dgl: 0.5.3

## Dataset Preprocessing

The raw datasets are included in `.txt` files (e.g. `Datasets/Video/Video.txt`), where each line contains a user id and item id. To preprocess the dataset, you could first generate dataset information:

```shell
python Datasets/Video/dataset_video.py
```

Then you could preprocess the dataset with the information:

```shell
python Datasets/Video/input_gen_video.py
```

## Run

To train our model on Video:

```shell
python ARBRE/Runs/run.py --data_name video --model_name arbre
```

