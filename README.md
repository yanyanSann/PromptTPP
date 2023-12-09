
# PromptTPP

Pytorch implementation for [Prompt-augmented Temporal Point Process for Streaming Event Sequence](https://arxiv.org/abs/2310.04993), NeurIPS 2023.


## How to Run
### Environment Requirements

First, please make sure you have an environment compatible with the following requirement 

```bash
torch == 1.11.0
numpy
pandas
```

Lower version of pytorch should also be working but we have not tested it. 


### Training and Evaluation Example

Assume we are running PT-attNHP over the Amazon data and setup the config files.

Step 1: We need to configure the parameter file corresponding to the dataset
```
vim dataset_config.yaml
```
NOTE: in `example_config/dataset_config.yaml`, one needs to setup information of the dataset, where we have put the default params of Amazon there.


Step 2: we need to choose the TPP model and configure the parameter file corresponding to the model
```
vim model_config.yaml
```
NOTE: in `example_config/model_config.yaml`, one needs to setup information of the model specs, where we have put the default params of PT-attNHP there.


Step 3: Then we train the chosen TPP model and evaluate

```
python run_pt_anhp.py
```


## Citing


If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@inproceedings{xue2023prompt,
  title={Prompt-augmented Temporal Point Process for Streaming Event Sequence},
  author={Xue, Siqiao and Wang, Yan and Chu, Zhixuan and Shi, Xiaoming and Jiang, Caigao and Hao, Hongyan and Jiang, Gangwei and Feng, Xiaoyun and Zhang, James Y and Zhou, Jun},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
  url={https://arxiv.org/abs/2310.04993}
}
```

## Credits

The following repositories are used in our code, either in close to original form or as an inspiration:

- [EasyTemporalPointProcess](https://github.com/ant-research/EasyTemporalPointProcess)
- [Neural Hawkes Process](https://github.com/hongyuanmei/neurawkes)
- [HYPRO TPP](https://github.com/ant-research/hypro_tpp)
