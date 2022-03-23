'''
Pol: This code is used to find the best set of images transformations for the BarlowTwins 
model applied to Chest X-Ray images. 

We are going to take into consideration the following transformations, extracted from the papers:
https://arxiv.org/pdf/2101.04909.pdf
https://arxiv.org/pdf/2006.13276.pdf
https://arxiv.org/pdf/2101.05224.pdf

##################################################
#### (1) random resizing/cropping
 The cropping from the random resizing/cropping augmentation was done at an image scale uniformly 
 distributed between 20% and 100% the size of the original image.
 Google paper: random crop to 224×224 pixels

#### (2) random horizontal flipping, 

#### (3) random vertical flipping, 

#### (4) random Gaussian blur, 
For the blur augmentation, we applied the following normalized Gaussian kernel: g(x, y) = 1 σkernel 
 √ 2π exp  − 1 2 x 2 + y 2 σ 2 kernel  , (3) where σ was selected for each sample uniformly at random 
 between 0.1 and 2.0 pixels.

#### (5) Gaussian noise addition, 
We selected the standard deviation for the noise addition randomly according to the following formula: 
 σnoise = µimage SNR , (4) where SNR was selected uniformly between 4 and 8 for each sample and µimage 
 was the average pixel value of the input sample image

#### (6) histogram normalization.

#### (7) random rotation 
by angle δ ∼ U(−20, 20) degree

#### (8) random additive brightness modulation
 Random additive brightness modulation adds a δ ∼ U(−0.2, 0.2) to all channels

#### (9) random multiplicative contrast modulation
 Random multiplicative contrast modulation multiplies per-channel standard deviation by a factor s ∼ U(−0.2, 0.2)

#### (10) change of perspective

##################################################
Due to the ammount of possibilities we have decided to proceed according to the following strategy:
1. We start with a set of "safe" transformations:
    - random resizing/cropping
    - random horizontal flipping, 

2. To this set of "safe" tranformations we will do one extra tranformation and train the model,
and we will iterate through all the extra transformations. Set of extra transformations:
    - random Gaussian blur + Gaussian noise addition
    - histogram normalization
    - random rotation
    - random additive brightness modulation + random multiplicative contrast modulation
    - change of perspective
    - random vertical flipping

3. Once done thnis, we will try combinations of the "safe" transformations + the most successful "extra"
transformations. 

##################################################
'''