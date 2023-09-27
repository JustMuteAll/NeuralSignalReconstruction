import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor
import torchvision.models as models
import torch.nn.functional as F

# Pretrained model settings
alexnet = models.alexnet(pretrained=True)

# Create a new model that includes all layers up to and including conv2/conv4/fc6
alexnet_conv2 = torch.nn.Sequential(*list(alexnet.features)[:5])
true_feature = torch.zeros((50,192,27,27))

# Compute the feature of original images
for i in range(50):
    img = Image.open("./Ori_to_compare/Ori_{}.jpg".format(i+1))
    img = ToTensor()(img)
    batch_img = torch.unsqueeze(img, 0).repeat(1, 3, 1, 1)
    # print(batch_img.shape)
    # Pass the image through the new model to get the output of fc6
    conv2_output = alexnet_conv2(batch_img)
    true_feature[i-1] = conv2_output
print(true_feature.shape)

# Compute the feature of IT reconstruction images
IT_feature = torch.zeros((50,192,27,27))
for i in range(50):
    img = Image.open("./IT_to_compare_true/IT__{}.jpg".format(i+1))
    img = ToTensor()(img)
    #print(img.shape)
    batch_img = torch.unsqueeze(img, 0).repeat(1, 1, 1, 1)
    # print(batch_img.shape)
    # Pass the image through the new model to get the output of fc6
    conv2_output = alexnet_conv2(batch_img)
    IT_feature[i-1] = conv2_output
print(IT_feature.shape)

# Compute PSM for IT images
count = np.zeros((50,1))
for i in range(50):
    ori_feature = torch.mean(true_feature[i],dim=0).flatten()
    norm_ori_feature = (ori_feature - torch.mean(ori_feature))/torch.std(ori_feature)
    rec_feature = torch.mean(IT_feature[i],dim=0).flatten()
    norm_rec_feature = (rec_feature - torch.mean(rec_feature))/torch.std(rec_feature)
    # print(norm_ori_feature.shape,norm_rec_feature.shape)
    baseline = F.cosine_similarity(norm_ori_feature.unsqueeze(0), norm_rec_feature.unsqueeze(0))
    for j in range(50):
        if j != i:
            others_feature = torch.mean(IT_feature[j],dim=0).flatten()
            norm_others_feature = (others_feature - torch.mean(others_feature))/torch.std(others_feature)
            #compute the pearson correlation coefficient between two images
            corr_others = F.cosine_similarity(norm_ori_feature.unsqueeze(0), norm_others_feature.unsqueeze(0))
            if corr_others < baseline:
                count[i] += 1
count = count/49
IT_early_PSM = np.mean(count)
IT_early_PSM_std = np.std(count)
print(IT_early_PSM,IT_early_PSM_std)