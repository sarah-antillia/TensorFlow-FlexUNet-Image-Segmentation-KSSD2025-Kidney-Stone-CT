<h2>TensorFlow-FlexUNet-Image-Segmentation-KSSD2025-Kidney-Stone-CT (2026/02/10)</h2>
Sarah T.  Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Kidney-Stone-CT</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass), 
and 
<a href="https://www.kaggle.com/datasets/murillobouzon/kssd2025-kidney-stone-segmentation-dataset">
<b>KSSD2025 - Kidney Stone Segmentation Dataset</b> </a> on the kaggle.com.
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of the original <b>KSSD2025</b> dataset, which contains 838 annotated images, 
we used our offline augmentation tool <a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a> (please see also: 
<a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a>)
 to generate a PNG <a href="https://drive.google.com/file/d/13F-R3-x2s9Gte2qrj_Wc556mAwJbojUS/view?usp=sharing">
  Augmented-KSSD2025-Kidney-Stone-CT-ImageMask-Dataset
 </a>.
<br><br> 
<hr>
<b>Actual Image Segmentation for Kidney-Stone-CT Images of 512x512 pixels </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/images/10110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/masks/10110.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test_output/10110.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/images/10122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/masks/10122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test_output/10122.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/images/10323.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/masks/10323.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test_output/10323.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://www.kaggle.com/datasets/murillobouzon/kssd2025-kidney-stone-segmentation-dataset">
<b>KSSD2025 - Kidney Stone Segmentation Dataset</b> </a>
<br><b>Dataset for automatic kidney stone segmentation in axial CT scans</b>
<br><br>
For more information, please refer to <a href="https://ieeexplore.ieee.org/document/11165055">
KSSD2025: A New Annotated Dataset for Automatic Kidney Stone Segmentation and Evaluation With Modified U-Net Based Deep Learning Models</a>
<br><br>
The following explanation was taken from 
<a href="https://www.kaggle.com/datasets/murillobouzon/kssd2025-kidney-stone-segmentation-dataset">KSSD2025</a> on the kaggle.com.
<br><br>
<b>About Dataset</b><br>
<b> Overview</b><br>
KSSD2025 is a dataset of axial CT images with expert-annotated kidney stone segmentation masks, created to support deep learning research in medical image segmentation. It is derived from the public dataset by Islam et al. (2022), which contains CT images with different kidney conditions. 
KSSD2025 focuses exclusively on kidney stone cases, offering precise ground-truth masks for developing and benchmarking AI-based segmentation models.
<br><br>
<b>Description</b><br>
This dataset presents a carefully refined subset of the original "CT Kidney Dataset: Normal-Cyst-Tumor and Stone" by Islam et al., comprising only 
axial CT images that exhibit kidney stones. Out of 12,446 images in the original collection, 
838 images were selected for manual annotation based on the presence of stones and the axial orientation, which 
offers better anatomical context for segmentation tasks.
<br><br>
<b>Dataset Details</b><br>
Total Annotated Images: 838<br>
View: Axial<br>
Annotations: Binary segmentation masks (kidney stone regions)<br>
Image Format: TIF<br>
Size: 305.38 MB<br>
Source Dataset: <a href="https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone">
CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone</a><br>
Annotation Method: Semi-automatic (thresholding + connected components) followed by expert manual refinement
<br><br>

<b>Citations</b><br>
Islam MN, Hasan M, Hossain M, Alam M, Rabiul G, Uddin MZ, Soylu A. Vision transformer and explainable transfer learning models 
for auto detection of kidney cyst, stone and tumor from CT-radiography. Scientific Reports. 2022.<br>
M. F. Bouzon et al., "KSSD2025: A New Annotated Dataset for Automatic Kidney Stone Segmentation and Evaluation 
with Modified U-Net Based Deep Learning Models," in IEEE Access, doi: 10.1109/ACCESS.2025.3610027
<br><br>
<b>License</b><br>
Datafiles © Nazmul Islam<br><br>
<b>Institutions Involved</b><br>
Centro Universitário FEI<br>
Hospital Universitário da Universidade de São Paulo<br>
Based on original dataset by:<br>
Islam MN, Hasan M, Hossain M, Alam M, Rabiul G, Uddin MZ, Soylu A. <br>
Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor 
from CT-radiography. Scientific Reports. 2022.
<!--
Please see also: <a href="https://ieeexplore.ieee.org/document/11165055">KSSD2025: A New Annotated Dataset for Automatic Kidney Stone Segmentation and Evaluation With Modified U-Net Based Deep Learning Models
</a>
-->
<br>
<br>
<h3>
2 Kidney-Stone-CT ImageMask Dataset
</h3>
 If you would like to train this Kidney-Stone-CT Segmentation model by yourself,
please down load  the <a href="https://drive.google.com/file/d/13F-R3-x2s9Gte2qrj_Wc556mAwJbojUS/view?usp=sharing">
  Augmented-KSSD2025-Kidney-Stone-CT-ImageMask-Dataset.zip
 </a> on the google driive.
</b><br>
We used our offline augmentation tool <a href="./generator/ImageMaskDataset.py">ImageMaskDataset.py</a> to generate 
the  Augmented Kidney-Stone-CT dataset.<br>
<pre>
./dataset
└─Kidney-Stone-CT
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Kidney-Stone-CT Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/Kidney-Stone-CT_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Kidney-Stone-CT TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Kidney-Stone-CT and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 2
base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Kidney-Stone-CT 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Kidney-Stone-CT 1+1
rgb_map = {(0,0,0):0, (255,255,255):1,  }       
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middle-point (13,14,15)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (28,29,30)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was terminated at epoch 30.<br><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/train_console_output_at_epoch30.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Kidney-Stone-CT</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Kidney-Stone-CT.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/evaluate_console_output_at_epoch30.png" width="880" height="auto">
<br><br>Image-Segmentation-Kidney-Stone-CT

<a href="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Kidney-Stone-CT/test was very low, and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0005
dice_coef_multiclass,0.9997
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Kidney-Stone-CT</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Kidney-Stone-CT.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Kidney-Stone-CT  Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/images/10118.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/masks/10118.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test_output/10118.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/images/10133.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/masks/10133.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test_output/10133.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/images/10172.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/masks/10172.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test_output/10172.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/images/10323.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/masks/10323.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test_output/10323.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/images/10348.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/masks/10348.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test_output/10348.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/images/10364.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test/masks/10364.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-CT/mini_test_output/10364.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. KSSD2025: A New Annotated Dataset for Automatic Kidney Stone Segmentation and Evaluation With Modified U-Net Based Deep Learning Models</b><br>
Murillo F. Murillobouzon; Paulo H. S. de Santana; Gabriel N. Missima; Weverson S. Pereira; Fernando P. Rivera; Gilson A. Giraldi<br>
<a href="https://ieeexplore.ieee.org/document/11165055">https://ieeexplore.ieee.org/document/11165055</a>
<br><br>
<b>2. A deep learning system for automated kidney stone detection and volumetric segmentation on noncontrast CT scans</b><br>
Daniel C. Elton, Evrim B.Turkbey, Perry J.Pickhardt, Ronald M.Summers<br>
<a href="https://www.moreisdifferent.com/assets/my_papers/B_AI_medical_imaging/2022_Z_Elton_Medical_Physics_kidney_stone_detector.pdf">
https://www.moreisdifferent.com/assets/my_papers/B_AI_medical_imaging/2022_Z_Elton_Medical_Physics_kidney_stone_detector.pdf</a>
<br><br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
