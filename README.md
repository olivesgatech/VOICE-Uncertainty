# VOICE-Uncertainty
[J-STSP Special Issue] VOICE: Variance of Induced Contrastive Explanations to quantify Uncertainty in Neural Network Interpretability

Official code repository for the paper: M. Prabhushankar and G. AlRegib, "VOICE: Variance of Induced Contrastive Explanations to quantify Uncertainty in Neural Network Interpretability," in IEEE Journal of Selected Topics in Signal Processing, doi: 10.1109/JSTSP.2024.3413536

Work conducted at [OLIVES@GaTech](https://alregib.ece.gatech.edu). Arxiv paper available at [https://arxiv.org/pdf/2406.00573](https://arxiv.org/pdf/2406.00573).

![Concept image showcasing uncertainty of GradCAM on VGG-16 and Swin Transformer Architectures](Figs/Concept.png)
GradCAM explanations and the proposed uncertainty visualization. Top row are results obtained on VGG-16 while bottom row are results obtained on Swin Transformer. Figs. (a), (b), (e), and (f) are obtained on a clean image where both VGG-16 and Swin Transformer predict correctly while Figs. (c), (d), (g), and (h) are results on noisy image where both networks predict incorrectly.

## Abstract
In this paper, we visualize and quantify the predictive uncertainty of gradient-based post hoc visual explanations for neural networks. Predictive uncertainty refers to the variability in the network predictions under perturbations to the input. Visual post hoc explainability techniques highlight features within an image to justify a network's prediction. We theoretically show that existing evaluation strategies of visual explanatory techniques partially reduce the predictive uncertainty of neural networks. This analysis allows us to construct a plug in approach to visualize and quantify the remaining predictive uncertainty of any gradient-based explanatory technique. We show that every image, network, prediction, and explanatory technique has a unique uncertainty. The proposed uncertainty visualization and quantification yields two key observations. Firstly, oftentimes under incorrect predictions, explanatory techniques are uncertain about the same features that they are attributing the predictions to, thereby reducing the trustworthiness of the explanation. Secondly, objective metrics of an explanation's uncertainty, empirically behave similarly to epistemic uncertainty. We support these observations on two datasets, four explanatory techniques, and six neural network architectures.

## Usage
We provide demo.py code that generates the explanation map, contrast map, and VOICE uncertainty map for three explanatory techniques:

<ol>
  <li>GradCAM</li>
  <li>GradCAM++</li>
  <li>SmoothGrad</li>
</ol>

Once you run demo.py, the following folder structure will be created with the .png images:

```
- Results
    - GradCAM
      - Explanation.png
      - Contrast.png
      - Uncertainty of Explanation.png
      - stats.txt
    - GradCAM++
      - Explanation.png
      - Contrast.png
      - Uncertainty of Explanation.png
      - stats.txt
    - SmoothGrad
      - Explanation.png
      - Contrast.png
      - Uncertainty of Explanation.png
      - stats.txt
```
stats.txt holds the following values:

```
stats = {
            'Prediction': pred_class,
            'Contrast Class': contrast_class,
            'SNR': snr,
            'Log Likelihood': log_likelihood,
            'IoU': iou,
            'Logit': logit,
    }
```

Variations in usage:

<ol>
  <li>Architectural change: Currently, the code is tested on pretrained models from [PyTorch's Torchvision Library](https://pytorch.org/vision/stable/models.html). GradCAM and GradCAM++ require target layers, defined in utils_GradCAM.py. Following network layers are defined: VGG16 (Default, Lines 177-178), SqueezeNet (Lines 180-181), AlexNet (Lines 183-184), DenseNet (Lines 186-187), Swin Transformer (Lines 189-191) and ResNet (Lines 193-194)</li>
  <li>Dataset: The models chosen above are pretrained on ImageNet. Hence, num_classes in line 197 = 1000. Please hange this variable accordingly.</li>
  <li>IoU Threshold: The IoU statsistic is dependent on threshold of the explanation and uncertainty heatmap. Currently it is set in Line 17. Please change this according to your data and application constraints.</li>
</ol>

## Citation

If you find this work useful, please cite the following two papers

[1] M. Prabhushankar and G. AlRegib, "VOICE: Variance of Induced Contrastive Explanations to quantify Uncertainty in Neural Network Interpretability," in IEEE Journal of Selected Topics in Signal Processing, doi: 10.1109/JSTSP.2024.3413536.

```
@ARTICLE{10555129,
  author={Prabhushankar, Mohit and AlRegib, Ghassan},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={VOICE: Variance of Induced Contrastive Explanations to quantify Uncertainty in Neural Network Interpretability}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  keywords={Uncertainty;Visualization;Neural networks;Transformers;Perturbation methods;Training;Task analysis;Predictive Uncertainty;Gradients;Contrastive explanations;Counterfactual explanations;Neural Networks;Deep Learning},
  doi={10.1109/JSTSP.2024.3413536}}
```

[2] M. Prabhushankar, G. Kwon, D. Temel and G. AlRegib, "Contrastive Explanations In Neural Networks," 2020 IEEE International Conference on Image Processing (ICIP), Abu Dhabi, United Arab Emirates, 2020, pp. 3289-3293, doi: 10.1109/ICIP40778.2020.9190927.

```
@INPROCEEDINGS{9190927,
  author={Prabhushankar, Mohit and Kwon, Gukyeong and Temel, Dogancan and AlRegib, Ghassan},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)}, 
  title={Contrastive Explanations In Neural Networks}, 
  year={2020},
  volume={},
  number={},
  pages={3289-3293},
  keywords={Visualization;Neural networks;Manifolds;Image recognition;Image quality;Automobiles;Image color analysis;Interpretability;Gradients;Deep Learning;Fine-Grained Recognition;Image Quality Assessment},
  doi={10.1109/ICIP40778.2020.9190927}}
```

## Questions?

If you have any questions, regarding the dataset or the code, you can contact the authors [(mohit.p@gatech.edu)](mohit.p@gatech.edu), or even better open an issue in this repo and we'll do our best to help.

## Code acknowledgements
The code is built on Grad-CAM. We use the implementation of https://github.com/1Konny/gradcam_plus_plus-pytorch as our base code. Specifically, gradcam.py, utils_gradcam.py, and utils.py are adapted. 
