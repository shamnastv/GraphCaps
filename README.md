PyTorch implementation of [**Capsule Graph Neural Network (ICLR 2019).**](https://openreview.net/forum?id=Byl8BnRcYm)


Basic capsule structures are taken from [[here]](https://github.com/timomernick/pytorch-capsule).

#### Data Preprocessing

Preprocessing program to generate specific experimental data format is taken from [original implementation](https://github.com/XinyiZ001/CapsGNN). The default raw data format should be `.gexf` (avalaible at [[gexf Dataset]](https://drive.google.com/drive/folders/1qXx-OZlJtgRYn579aQX13ou2hutqJz41?usp=sharing)). Each line of the label file represents a graph with the format <br/>
```
    xxx.gexf label
```
To generate experimental data format:

Extract dataset in graph_gexf/ and run

```
    $ python3 preprocessing.py --dataset_input_dir graph_gexf/ENZYMES
```    
Structure of graph should be mentioned in graph_structure.json

#### Execute

```
    $ python3 main.py --dataset_dir data_plk/ENZYMES
```

