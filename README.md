# VOICE-Uncertainty
[J-STSP Special Issue] VOICE: Variance of Induced Contrastive Explanations to quantify Uncertainty in Neural Network Interpretability

Official code repository for the paper: M. Prabhushankar and G. AlRegib, "VOICE: Variance of Induced Contrastive Explanations to Quantify Uncertainty in Neural Network Interpretability," Journal of Selected Topics in Signal Processing (J-STSP) Special Series on AI in Signal & Data Science - Towards Explainable, Reliable, and Sustainable Machine Learning, May 23, 2024.

![Concept image showcasing uncertainty of GradCAM on VGG-16 and Swin Transformer Architectures](Figs/Concept.png)
GradCAM explanations and the proposed uncertainty visualization. Top row are results obtained on VGG-16 while bottom row are results obtained on Swin Transformer. Figs. (a), (b), (e), and (f) are obtained on a clean image where both VGG-16 and Swin Transformer predict correctly while Figs. (c), (d), (g), and (h) are results on noisy image where both networks predict incorrectly.

## Abstract
In this paper, we visualize and quantify the predictive uncertainty of gradient-based post hoc visual explanations for neural networks. Predictive uncertainty refers to the variability in the network predictions under perturbations to the input. Visual post hoc explainability techniques highlight features within an image to justify a network's prediction. We theoretically show that existing evaluation strategies of visual explanatory techniques partially reduce the predictive uncertainty of neural networks. This analysis allows us to construct a plug in approach to visualize and quantify the remaining predictive uncertainty of any gradient-based explanatory technique. We show that every image, network, prediction, and explanatory technique has a unique uncertainty. The proposed uncertainty visualization and quantification yields two key observations. Firstly, oftentimes under incorrect predictions, explanatory techniques are uncertain about the same features that they are attributing the predictions to, thereby reducing the trustworthiness of the explanation. Secondly, objective metrics of an explanation's uncertainty, empirically behave similarly to epistemic uncertainty. We support these observations on two datasets, four explanatory techniques, and six neural network architectures.

## Questions?

If you have any questions, regarding the dataset or the code, you can contact the authors [(mohit.p@gatech.edu)](mohit.p@gatech.edu), or even better open an issue in this repo and we'll do our best to help.

