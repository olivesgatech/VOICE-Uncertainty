import os
import PIL
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
import json

from utils import visualize_cam, Normalize
from utils_GradCAM import convert_to_grayscale
from gradcam import GradCAM, GradCAMpp, Contrast, Contrast_pp, GuidedBackprop, generate_smooth_grad, guided_grad_cam, GuidedBackprop_Contrast

def iou_numpy(outputs: np.array, labels: np.array):

    threshold = 0.1
    outputs = outputs.numpy()
    outputs = np.where(outputs > threshold, 1, 0)
    labels = np.where(labels > threshold, 1, 0)

    SMOOTH = 1e-6

    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def get_stats(explanation, uncertainty, logit, label):

    snr = signaltonoise(uncertainty.numpy(), axis=None)
    target = torch.from_numpy(np.asarray([label])).cuda()
    log_likelihood = -F.nll_loss(logit, target).data.cpu().numpy()
    iou = iou_numpy(uncertainty, explanation)

    return iou, snr, log_likelihood

def get_explanation(explanation_model, contrast_model, img, num_classes):


    mask_explanation, logit = explanation_model(img)
    mask_explanation = mask_explanation.squeeze(0).squeeze(0).cpu().numpy()

    _, classes = torch.topk(logit, 2)
    classes = classes.squeeze(0)

    pred_class = classes[0].data.cpu().numpy()
    contrast_class = classes[1].data.cpu().numpy()

    mask_contrast, _ = contrast_model(img, contrast_class)
    mask_contrast = mask_contrast.squeeze(0).squeeze(0).cpu().numpy()

    uncertain_mask = []
    for num_class in range(num_classes):

        # print(curr_class)
        if num_class == pred_class:
            continue

        mask_temp, _ = contrast_model(img, num_class)  # , retain_graph = True)
        mask_temp = mask_temp.squeeze(0).squeeze(0).data.cpu()
        uncertain_mask.append(mask_temp)

        del mask_temp

    uncertain_mask_all = torch.stack(uncertain_mask, 0)
    mask_uncertainty = torch.var(uncertain_mask_all.data, 0)

    saliency_map_min, saliency_map_max = mask_uncertainty.min(), mask_uncertainty.max()
    mask_uncertainty = (mask_uncertainty - saliency_map_min).div(saliency_map_max - saliency_map_min).data

    iou, snr, log_likelihood = get_stats(mask_explanation, mask_uncertainty, logit, pred_class)
    stats = {
            'Prediction': pred_class,
            'Contrast Class': contrast_class,
            'SNR': snr,
            'Log Likelihood': log_likelihood,
            'IoU': iou,
            'Logit': logit,
    }

    return mask_explanation, mask_contrast, mask_uncertainty, stats

def get_explanation_smoothgrad(explanation_model, contrast_model, img, pred, contrast_class, logit, num_classes):

    param_n = 10
    param_sigma_multiplier = 5

    smooth_grad = generate_smooth_grad(explanation_model,
                                       img,
                                       pred,
                                       param_n,
                                       param_sigma_multiplier)
    grayscale_guided_grads = convert_to_grayscale(smooth_grad)
    mask_explanation = grayscale_guided_grads.squeeze(0)

    smooth_grad_contrast = generate_smooth_grad(contrast_model,
                                       img,
                                       contrast_class,
                                       param_n,
                                       param_sigma_multiplier)
    grayscale_guided_grads = convert_to_grayscale(smooth_grad_contrast)
    mask_contrast = grayscale_guided_grads.squeeze(0)

    uncertain_mask = []
    for num_class in range(num_classes):

        if num_class == pred:
            continue

        mask_gbp_c = generate_smooth_grad(contrast_model,
                                       img,
                                       num_class,
                                       param_n,
                                       param_sigma_multiplier)
        grayscale_guided_grads_c = convert_to_grayscale(mask_gbp_c).squeeze(0)
        uncertain_mask.append(torch.Tensor(grayscale_guided_grads_c))

        del grayscale_guided_grads_c, mask_gbp_c

    uncertain_mask_all = torch.stack(uncertain_mask, 0)
    mask_uncertainty = torch.var(uncertain_mask_all.data, 0)

    saliency_map_min, saliency_map_max = mask_uncertainty.min(), mask_uncertainty.max()
    mask_uncertainty = (mask_uncertainty - saliency_map_min).div(saliency_map_max - saliency_map_min).data

    iou, snr, log_likelihood = get_stats(mask_explanation, mask_uncertainty, logit, pred)
    stats = {
        'Prediction': pred,
        'Contrast Class': contrast_class,
        'SNR': snr,
        'Log Likelihood': log_likelihood,
        'IoU': iou,
        'Logit':logit,
    }

    return mask_explanation, mask_contrast, mask_uncertainty, stats

def save_explanations(explanation, contrast, uncertainty, image, output_dir):

    _, result = visualize_cam(explanation, image)
    output_path = os.path.join(output_dir + 'Explanation.png')
    save_image(result, output_path)

    _, result_contrast = visualize_cam(contrast, image)
    output_path = os.path.join(output_dir + 'Contrast.png')
    save_image(result_contrast, output_path)

    _, result = visualize_cam(uncertainty, image)
    output_path = os.path.join(output_dir + 'Uncertainty of Explanation.png')
    save_image(result, output_path)

def main():

    img_dir = 'images'
    #img_name = 'water-bird.JPEG'
    img_name = 'cat_dog.png'
    img_path = os.path.join(img_dir, img_name)
    pil_img = PIL.Image.open(img_path)

    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)

    normed_torch_img = normalizer(torch_img)


    #Choose Architecture
    arch = models.vgg16(pretrained=True)
    model_dict = dict(type='vgg', arch=arch, layer_name='features_29', input_size=(224, 224))

    #arch = models.squeezenet1_0(pretrained=True)
    #model_dict = dict(type='squeezenet', arch=arch, layer_name='features_12_expand3x3_activation', input_size=(224, 224))

    #arch = models.alexnet(pretrained=True)
    #model_dict = dict(type='alexnet', arch=arch, layer_name='features_11', input_size=(224, 224))

    #arch = models.densenet169(pretrained=True)
    #model_dict = dict(type='densenet', arch=arch, layer_name='features_norm5', input_size=(224, 224))

    #arch = models.swin_b(pretrained=True)
    #target_layer = [arch.features[-2].norm]
    #model_dict = dict(type='swin', arch=arch, layer_name=target_layer, input_size=(224, 224))

    #arch = models.resnet18(pretrained=True)
    #model_dict = dict(type='resnet', arch=arch, layer_name='layer4', input_size=(224, 224))


    num_classes = 1000 #Number of classes in ImageNet

    #GradCAM
    print('Computing GradCAM and its VOICE uncertainty')
    gradcam = GradCAM(model_dict, False)
    contrast = Contrast(model_dict, False)

    explanation_map, contrast_map, uncertainty_map, stats = get_explanation(gradcam, contrast, normed_torch_img, num_classes)

    output_dir = 'Results/GradCAM/'
    os.makedirs(output_dir, exist_ok=True)

    save_explanations(explanation_map, contrast_map, uncertainty_map, torch_img, output_dir)
    torch.save(stats, output_dir + 'stats_gradcam.t7')
    with open(output_dir + 'stats.txt', 'w') as file:
        file.write(str(stats))
    print('GradCAM complete')


    #GradCAM++
    print('Computing GradCAM++ and its VOICE uncertainty')
    gradcampp = GradCAMpp(model_dict, False)
    contrastpp = Contrast_pp(model_dict, False)

    explanation_map, contrast_map, uncertainty_map, stats = get_explanation(gradcampp, contrastpp, normed_torch_img, num_classes)

    output_dir = 'Results/GradCAM++/'
    os.makedirs(output_dir, exist_ok=True)

    save_explanations(explanation_map, contrast_map, uncertainty_map, torch_img, output_dir)
    torch.save(stats, output_dir + 'stats_gradcam++.t7')
    with open(output_dir + 'stats.txt', 'w') as file:
        file.write(str(stats))
    print('GradCAM++ complete')


    #SmoothGrad
    print('Computing SmoothGrad and its VOICE uncertainty')
    pred = stats['Prediction']
    contrast_class = stats['Contrast Class']
    logit = stats['Logit']
    GBP = GuidedBackprop(arch)
    GBP_c = GuidedBackprop_Contrast(arch)

    explanation_map, contrast_map, uncertainty_map, stats = get_explanation_smoothgrad(GBP, GBP_c, normed_torch_img, pred, contrast_class, logit, num_classes)

    output_dir = 'Results/SmoothGrad/'
    os.makedirs(output_dir, exist_ok=True)

    save_explanations(explanation_map, contrast_map, uncertainty_map, torch_img, output_dir)
    torch.save(stats, output_dir + 'stats_sm.t7')
    with open(output_dir + 'stats_sm.txt', 'w') as file:
        file.write(str(stats))
    print('SmoothGrad complete')


if __name__ == '__main__':
    main()
