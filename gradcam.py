import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import os
from torchvision.utils import make_grid, save_image
import PIL
import math
import torchvision.models as models
from torch.nn import ReLU


from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from itertools import chain, combinations

from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer, find_resnet18_layer, find_swin_layer

def reshape_transform(tensor, height=7, width=7):

    result = tensor
    result = result.transpose(2, 3).transpose(1, 2)

    return result

class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'small' in model_type.lower():
            target_layer = find_resnet18_layer(self.model_arch, layer_name)
        elif 'swin' in model_type.lower():
            target_layer = model_dict['layer_name'][0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                #print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        self.model_arch.eval()
        self.model_arch.cuda()
        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()
        #print(logit.max(1))
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']

        #gradients = reshape_transform(gradients)
        #activations = reshape_transform(activations)

        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        self.model_arch.cuda()
        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        #ce_loss = nn.CrossEntropyLoss()
        #class_idx = 242

        #im_label_as_var = Variable(torch.from_numpy(np.asarray([class_idx])))
        #pred_loss = ce_loss(logit.cuda(), im_label_as_var.cuda())
        #score = pred_loss

        self.model_arch.zero_grad()
        score.backward()#retain_graph=retain_graph)
        gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A

        #gradients = reshape_transform(gradients)
        #activations = reshape_transform(activations)

        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit

class Contrast_pp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        super(Contrast_pp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        self.model_arch.cuda()
        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()



        ce_loss = nn.CrossEntropyLoss()
        #class_idx = 242

        im_label_as_var = Variable(torch.from_numpy(np.asarray([class_idx])))
        pred_loss = ce_loss(logit.cuda(), im_label_as_var.cuda())
        score = pred_loss

        logit = F.relu(logit)

        '''
        score_q = -logit[:, class_idx].squeeze()
        score_p = logit[:, logit.max(1)[-1]].squeeze()
        score_q_1 = -logit[:, logit.min(1)[-1]].squeeze()

        #score = (0.1 * score_p).pow(2) + score_q
        score = (0.1 * score_p).pow(2) + score_q
        
        #self.model_arch.zero_grad()
        #score.backward(retain_graph=retain_graph)
        '''

        self.model_arch.zero_grad()
        score.backward()  # retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A

        #gradients = reshape_transform(gradients)
        #activations = reshape_transform(activations)

        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                      activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data



        return saliency_map, logit

class Contrast(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'curenet' in model_type.lower():
            target_layer = self.model_arch.conv2
        elif 'swin' in model_type.lower():
            target_layer = model_dict['layer_name'][0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        self.model_arch.eval()
        self.model_arch.cuda()
        input.cuda()
        logit = self.model_arch(input)
        ce_loss = nn.CrossEntropyLoss()


        x_max, _ = torch.max(logit.data, 1)
        x_max = x_max.unsqueeze(0)

        x = x_max.cpu().numpy()
        #print(x)

        im_label_as_var = Variable(torch.from_numpy(np.asarray([class_idx])))
        pred_loss = ce_loss(logit.cuda(), im_label_as_var.cuda())

        self.model_arch.zero_grad()
        pred_loss.backward(retain_graph=retain_graph)


        gradients = self.gradients['value']
        activations = self.activations['value']

        b, k, u, v = gradients.size()

        alpha = (gradients.view(b, k, -1).mean(2))
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False).data
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):

        '''
        output_dir = 'outputs'
        folder2 = 'Contrast'
        folder1 = 'GradCam'

        img = []
        for ii in range(10):

            print(ii)

            temp_img = self.forward(input, ii, retain_graph)
            temp_img = temp_img.squeeze()
            temp_img = temp_img.squeeze()

            output_path = output_dir + '/' + folder2 + '/' + str(ii) + '.png'
            save_image(temp_img, output_path)
            PIL.Image.open(output_path)

            img.append(temp_img)

        return img
        '''

        return self.forward(input, class_idx, retain_graph)


def generate_smooth_grad(Backprop, prep_img, target_class, param_n, param_sigma_multiplier):
    """
        Generates smooth gradients of given Backprop type. You can use this with both vanilla
        and guided backprop
    Args:
        Backprop (class): Backprop type
        prep_img (torch Variable): preprocessed image
        target_class (int): target class of imagenet
        param_n (int): Amount of images used to smooth gradient
        param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
    """
    # Generate an empty image/matrix
    smooth_grad = np.zeros(prep_img.size()[1:])

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).item()
    for x in range(param_n):
        # Generate noise
        noise = Variable(prep_img.data.new(prep_img.size()).normal_(mean, sigma**2))
        # Add noise to the image
        noisy_img = prep_img + noise

        noisy_img = Variable(noisy_img)

        noisy_img.requires_grad = True

        # Calculate gradients
        vanilla_grads = Backprop.generate_gradients(noisy_img, target_class, 0)
        # Add gradients to smooth_grad
        #vanilla_grads = np.vstack(vanilla_grads)

        smooth_grad = smooth_grad + vanilla_grads
        #smooth_grad[1] = smooth_grad[1] + vanilla_grads[1]
        #smooth_grad[2] = smooth_grad[2] + vanilla_grads[2]
    # Average it out
    smooth_grad = smooth_grad / param_n
    return smooth_grad

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    #if not os.path.exists('../results'):
    #    os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image

    save_image(gradient, file_name)

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        #for pos, module in self.model.features._modules.items():
        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class, control):
        # Forward pass
        b, c, h, w = input_image.size()
        if control ==1:
            input_image.requires_grad = True
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.cuda()
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        #gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        gradients_as_arr = input_image._grad.data.cpu().numpy()[0]


        #saliency = torch.Tensor(grayscale_guided_grads).unsqueeze(0)

        #saliency_map = F.upsample(saliency, size=(h, w), mode='bilinear', align_corners=False)
        #saliency_map = saliency_map.data.cpu().numpy() #[0]
        #saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        #saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        #pos_sal, neg_sal = get_positive_negative_saliency(grayscale_guided_grads)
        #save_gradient_images(torch.Tensor(pos_sal), 'presentation/Thesis/Defense/GradCam/GBP_pos.png')
        #save_gradient_images(torch.Tensor(neg_sal), 'presentation/Thesis/Defense/GradCam/GBP_neg.png')

        return gradients_as_arr#, pos_sal, neg_sal


class GuidedBackprop_Contrast():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        #for pos, module in self.model.features._modules.items():
        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class, control):
        # Forward pass
        b, c, h, w = input_image.size()
        if control ==1:
            input_image.requires_grad = True
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        #one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        #one_hot_output[0][target_class] = 1
        #one_hot_output = one_hot_output.cuda()

        ce_loss = nn.CrossEntropyLoss()
        im_label_as_var = Variable(torch.from_numpy(np.asarray([target_class])))
        pred_loss = ce_loss(model_output.cuda(), im_label_as_var.cuda())

        # Backward pass
        #self.model_arch.zero_grad()
        pred_loss.backward(retain_graph=False)

        #model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        #gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        gradients_as_arr = input_image._grad.data.cpu().numpy()[0]


        #saliency = torch.Tensor(grayscale_guided_grads).unsqueeze(0)

        #saliency_map = F.upsample(saliency, size=(h, w), mode='bilinear', align_corners=False)
        #saliency_map = saliency_map.data.cpu().numpy() #[0]
        #saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        #saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        #pos_sal, neg_sal = get_positive_negative_saliency(grayscale_guided_grads)
        #save_gradient_images(torch.Tensor(pos_sal), 'presentation/Thesis/Defense/GradCam/GBP_pos.png')
        #save_gradient_images(torch.Tensor(neg_sal), 'presentation/Thesis/Defense/GradCam/GBP_neg.png')

        return gradients_as_arr#, pos_sal, neg_sal