# FaceMemNet
The repo for FaceMemNet: Predicting Face Memorability with DeepNeural Networks paper.

In this work we provide various models for predecting the memorability of face Images. 
We have leveraged SeNet50, ResNet50 and VGG16 that were pretrained on VGGFace2 database. 

### Pretrained models
The pretrained models were obtained from: https://github.com/rcmalli/keras-vggface

### Models
 We have fine-tuned _VGG16_, _ResNet50_ and _SENet50_ and the combinations of their features for predicting memorability. [10k US Adult Face Databse](https://www.wilmabainbridge.com/facememorability2.html) was used as the memorability database. This database includes 2222 face images with their corresponding memorability scores. 
 
 Corrected hit rate and hit rate can be used for fine-tuning the models. We did it with either cases. The trained models can be found [models trained with hit rate](https://drive.google.com/drive/folders/1vs1BVzFPtX-wlZgVTLHk9pYkve1DSlLN?usp=sharing) and [models trained with corrected hit rate](https://drive.google.com/drive/folders/1sborFlT0Aq5nHLM8p7IYYK7FRi4M45_S?usp=sharing).
 
 ### Results
 
 The performance of the aformentioned models can be found in the following tables:
 
 | Model  | Hit Rate Score | Corrected Hit Rate Score |
| ------------- | ------------- | ------------- |
| VGG16  | 0.445  | 0.579  |
| ResNet50  | 0.433  | 0.607  |
| SENet50  | 0.448 | 0.601  |
| ResVGG  | 0.423 | 0.626  |
| SENRes  | 0.452 | 0.631 |
| SENVGG  | **0.468** | 0.605  |
| SENResVGG  | 0.445 | **0.634**  |

