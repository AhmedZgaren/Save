import imgaug.augmenters as iaa
import numpy as np



class ImgAugTransform():
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    self.aug = iaa.Sequential([
        
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Fliplr(0.5),

        iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(0.1, size_percent=0.5)])),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
        
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        # In some images move pixels locally around (with random
        # strengths).
        sometimes(
            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
        ),

        # In some images distort local areas with varying strength.
        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
    ], random_order=True)
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)
