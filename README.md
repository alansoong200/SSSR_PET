# PET image super-resolution using generative adversarial networks
Tzu-An Song<sup>1</sup>, Samadrita Roy Chowdhury<sup>1</sup>, Fan Yang<sup>1</sup>, Joyita Dutta<sup>1</sup></br>
<sup>1</sup>Department of Electrical and Computer Engineering, University of Massachusetts Lowell, Lowell, MA, 01854 USA and co-affiliated with Massachusetts General Hospital, Boston, MA, 02114.

The intrinsically low spatial resolution of positron emission tomography (PET) leads to image quality degradation and inaccurate image-based quantitation. Recently developed supervised super-resolution (SR) approaches are of great relevance to PET but require paired low- and high-resolution images for training, which are usually unavailable for clinical datasets. In this paper, we present a self-supervised SR (SSSR) technique for PET based on dual generative adversarial networks (GANs), which precludes the need for paired training data, ensuring wider applicability and adoptability. The SSSR network receives as inputs a low-resolution PET image, a high-resolution anatomical magnetic resonance (MR) image, spatial information (axial and radial coordinates), and a high-dimensional feature set extracted from an auxiliary CNN which is separately-trained in a supervised manner using paired simulation datasets. The network is trained using a loss function which includes two adversarial loss terms, a cycle consistency term, and a total variation penalty on the SR image. We validate the SSSR technique using a clinical neuroimaging dataset. We demonstrate that SSSR is promising in terms of image quality, peak signal-to-noise ratio, structural similarity index, contrast-to-noise ratio, and an additional no-reference metric developed specifically for SR image quality assessment. Comparisons with other SSSR variants suggest that its high performance is largely attributable to simulation guidance.

Published in: Neural Networks

Pages: 83 - 91

DOI: 10.1016/j.neunet.2020.01.029

The paper can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S0893608020300393?via%3Dihub).

Our previous work is also available [here](https://github.com/alansoong200/SR_PET_CNN) on github.


## Prerequisites

This code uses:

- Python 2.7
- Pytorch 0.4.0
- matplotlib 2.2.4
- numpy 1.16.4
- scipy 1.2.1
- NVIDIA GPU
- CUDA 8.0
- CuDNN 7.1.2

## Dataset

BrainWeb (Simulated Brain Database):
https://brainweb.bic.mni.mcgill.ca/brainweb/

Alzheimer’s Disease Neuroimaging Initiative (ADNI) (Clinical Database):
http://adni.loni.usc.edu/

## Citation
If this code inspire you, please cite the paper.

	@article{
		song_pet_2020,
		title = {{PET} image super-resolution using generative adversarial networks},
		volume = {125},
		issn = {0893-6080},
		url = {http://www.sciencedirect.com/science/article/pii/S0893608020300393},
		doi = {10.1016/j.neunet.2020.01.029},
		language = {en},
		urldate = {2020-07-02},
		journal = {Neural Networks},
		author = {Song, Tzu-An and Chowdhury, Samadrita Roy and Yang, Fan and Dutta, Joyita},
		month = may,
		year = {2020},
		keywords = {Super-resolution, CNN, GAN, Multimodality imaging, PET, Self-supervised},
		pages = {83--91},
	}
}

## UMASS_LOWELL_BIDSLab
Biomedical Imaging & Data Science Laboratory

Lab's website:
http://www.bidslab.org/index.html


Email: TzuAn_Song(at)student.uml.edu, 
       TzuAn.Song(at)MGH.HARVARD.EDU, 
       alansoong200(at)gamil.com.
