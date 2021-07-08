# Fine-grained Image Classification via Pytorch

Simple code based on Pytorch pre-trained Resnet50.

## Accuracy

| Dataset                                                      | w/o amp | apex.amp | torch.cuda.amp | SOTA                                                         |
| ------------------------------------------------------------ | ------- | -------- | -------------- | ------------------------------------------------------------ |
| [CUB-200-2011](https://github.com/cyizhuo/CUB-200-2011-dataset) | 86.74   | 86.68    | 86.59          | [91.7](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200) |
| [FGVC Aircraft](https://github.com/cyizhuo/FGVC-Aircraft-dataset) | 93.25   | 93.58    | 92.86          | [94.7](https://paperswithcode.com/sota/fine-grained-image-classification-on-fgvc) |
| [Stanford Cars](https://github.com/cyizhuo/Stanford-Cars-dataset) | 94.09   | 94.30    | 94.32          | [96.32](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford) |
