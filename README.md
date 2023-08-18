# Product defects detection using Deep Learning
Detect products' surface defects using Deep Learning


## Abstract
Defects can result in significant waste and expenses to manufacturing businesses, and detecting surface defects is a challenging task that has received considerable attention in recent decades. Traditional image processing techniques can address certain types of problems, but they struggle with complex backgrounds, noise, and lighting differences. Deep learning has recently emerged as a solution to these challenges, driven by advances in computing power and the availability of large datasets. This research paper proposes the use of deep learning models to detect surface flaws, with potential benefits for manufacturing businesses' quality assurance.


## Dataset Description
Northeastern University (NEU) surface defect database (*)
Author: Kechen Song and Yunhui Yan
Link to download: [Google Drive]([url](https://drive.google.com/file/d/1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k/view)) / Google Drive ([Backup link]([url](https://drive.google.com/file/d/1epWS-oQ6UsYCFhXDbc8EbXxpXJjpRhBT/view?usp=share_link)))

The dataset has 1,800 grayscale images with annotations: 300 samples each of six different kinds of typical surface defects of the hot-rolled steel strip: rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In) and scratches (Sc). The original resolution of each image is 200×200 pixels.

**Figure 1** shows some sample images of 6 kinds of typical surface defects:

<p align="center">
<img width="433" alt="Screenshot 2023-08-18 at 12 03 35 PM" src="https://github.com/panhh20/Defects_Detection/assets/122824839/302c0014-efb6-45f3-879c-19731e8d54af"> </p>

In short, the NEU surface defect database includes two difficult challenges, i.e., the intra-class defects existing large differences in appearance while the inter-class defects have similar aspects, the defect images suffer from the influence of illumination and material changes. The dataset includes annotations
which indicate the class and location of a defect in each image. For each defect, the yellow box is the bounding box indicating its location and the green label is the class score.

<p align="center">
<img align=”center” width="452" alt="Screenshot 2023-08-18 at 12 04 41 PM" src="https://github.com/panhh20/Defects_Detection/assets/122824839/e8febe11-fa29-44fe-aa2d-c898dc31e915" loc="center"> </p>


## Methodology
I chose Convolutional Neural Network (CNN) since it’s a specialized model for image and video processing. It has also been demonstrated to be an effective tool for automated identification of defects (Rahman, 2022).


I also proposed a model that integrated Recurrent Neural Network (RNN) and Convolution Neural Network (CNN). Recurrent neural network (RNN) is a special type of neural network that is well-suited for processing sequential data. This makes them ideal for defect detection, as defects are usually observed in pixel sequences. RNNs can also assist in accelerating flaws detection by “remembering” information in previous frames and use them to predict objects in future frames.
