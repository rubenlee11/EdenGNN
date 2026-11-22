# EdenGNN documentation

EdenGNN predicts charge density from atomic structures. The predicted density is then used in fixed-density DFT calculations to obtain electronic structures.

## VASP workflows

1. [Train a model](vasp-training.md): prepare SAD and SCF data, then train pseudo density and PAW augmentation occupancies.
2. [Predict with your model](vasp-prediction.md): predict PAW charge density and calculate bands and DOS.
3. [Use EdenGNN-Uni](vasp.md): predict electronic structures with the pretrained VASP models.

## ABACUS workflows

1. [Train a model](abacus-training.md): prepare SCF difference-density labels and train a density model.
2. [Predict with your model](abacus-prediction.md): predict difference density and calculate bands and DOS.
3. [Use EdenGNN-Uni](abacus.md): predict electronic structures with the pretrained ABACUS model.

## Getting started

See [Installation and setup](setup.md) before running a workflow. Keep calculation outputs outside the source checkout. Use consistent pseudopotentials, basis sets, and DFT settings for training and prediction.

Scripts and templates are installed under `~/.edengnn/`. In this guide, script paths are relative to that directory. Configuration parameters are described in the repository's `config.yaml`.

For agent-assisted operation, install [EdenGNN-Skill](https://github.com/rubenlee11/EdenGNN-Skill) and describe the desired workflow with `$edengnn`.

## Additional topics

- [ABACUS + PyATB optical properties](optics.md)
- [Troubleshooting](troubleshooting.md)
- [Build and publish the documentation](publishing.md)

## Resources

- [EdenGNN source](https://github.com/rubenlee11/EdenGNN)
- [EdenGNN skill](https://github.com/rubenlee11/EdenGNN-Skill)
- [Pretrained models](https://huggingface.co/TrueSavage/EdenGNN)
- [Example density data](https://huggingface.co/datasets/TrueSavage/EdenGNN-Data)
- [Colab tutorial](https://colab.research.google.com/drive/1tSGPZk4XI71GEylYeNDD1218smOEaFKc?usp=sharing)
