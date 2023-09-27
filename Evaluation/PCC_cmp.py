import numpy as np
from PIL import Image

# Set path of original images and reconstructed images
ori_path = r"C:\Users\DELL\Desktop\Reconstruction_result\Corr\Ori_to_compare"
rec_path = r"C:\Users\DELL\Desktop\Reconstruction_result\Corr\IT_to_compare_true"

count = 0
avg_baseline = 0

for i in range(50):
    # Read images
    rec_img = Image.open(rec_path + '\\IT__' + str(i+1) + '.jpg')
    rec_img = np.array(rec_img)[:,:,0]
    cor_img = Image.open(ori_path + '\\Ori_' + str(i+1) + '.jpg')
    cor_img = np.array(cor_img)
    # Compute the pearson correlation coefficient between two images
    baseline = np.corrcoef(rec_img.flatten(), cor_img.flatten())[0, 1]
    avg_baseline += baseline
    # Compare with other images,check whether the PCC is smaller than baseline
    for j in range(50):
        if j != i:
            ori_img = Image.open(ori_path + '\\Ori_' + str(j+1) + '.jpg')
            #compute the pearson correlation coefficient between two images
            ori_img = np.array(ori_img)
            corr_others = np.corrcoef(ori_img.flatten(), rec_img.flatten())[0, 1]
            if corr_others < baseline:
                count += 1

avg_count = count/50/49
avg_baseline = avg_baseline/50

print(avg_count) # average number of images whose PCC is smaller than baseline 
print(avg_baseline) # average PCC between original images and reconstructed images
