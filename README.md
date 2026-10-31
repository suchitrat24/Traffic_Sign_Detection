# Traffic Sign Detection Using CNN

## Project Overview

This project builds a deep learning model that looks at a picture of a traffic sign and recognizes **which of 43 sign types** it is (speed limit 50, stop, yield, no entry, and so on), using Convolutional Neural Networks (CNNs). This is the step a self-driving car or driver-assistance system needs after a camera has found a sign on the road.

The project was completed as part of the **DataMites AI Engineer Program (Internship Project)**.

The primary objective was to analyze what the traffic sign images look like (size, quality, colour, shape), build and compare several CNN models, choose the best one, and give practical suggestions on how to use the model in a real driver-assistance system.

The images come from the German Traffic Sign Recognition Benchmark (GTSRB): **51,839 images** (39,209 training and 12,630 test) across **43 classes**.

## Project Objectives

- Perform data quality checks and exploratory data analysis (EDA) on the images.
- Detect duplicate images and check for broken files and missing values.
- Identify and prevent data leakage between near-identical video frames.
- Analyze how image size, brightness, contrast, sharpness and sign shape/colour vary across classes.
- Build and compare a custom CNN with pretrained transfer learning models.
- Evaluate test-time augmentation and ensemble methods.
- Provide data-driven suggestions for using the model in a driver-assistance system.

## Technologies & Libraries

- Python
- Jupyter Notebook
- PyTorch
- Torchvision
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow (PIL)

## Project Workflow

### 1. Data Checks and Cleaning
- Confirmed that the CSV files and image folders match exactly (51,839 PNG files, every path exists, CSV width/height equal the real image size).
- All images are PNG and RGB, with no missing values and no broken files.
- Found that each physical sign was filmed as a **track of 30 frames** (1,307 tracks in training, all with 30 frames except one with 29). The smallest classes have only 7 tracks.
- Found **8 exact duplicates** using MD5 fingerprints, all being the same image in Train and Test. They were kept so the official test set stays complete (at most 0.06 points of test accuracy).
- Converted all images to RGB and resized them to **64 x 64** pixels.
- Added analysis features: shape, sign colour, aspect ratio, sign width, mean red/green/blue, brightness, contrast and sharpness.

### 2. Exploratory Data Analysis (EDA)
- Viewed sample images, the clean sign drawings, and a full track of one sign from far to near.
- Analyzed image size, brightness, contrast, sharpness and aspect ratio distributions.
- Examined class balance, sign shapes and colours, and per-class brightness, width and sharpness.
- Used histograms, relational plots, pair plots, box plots, line plots, bar charts and correlation heatmaps.

### 3. Outlier Analysis
Used the Interquartile Range (IQR) method and box plots to identify extreme values.

Key findings:
- 3,370 images (6.5%) are wider than 94 px. These are close-up frames at the end of a track, the clearest images in the data.
- 299 very bright images (0.6%) and 842 very sharp images (1.6%) were flagged. Contrast had no outliers.
- The darkest images (85 below brightness 10) look completely black, but after stretching the brightness the sign is clearly visible.
- All outliers were **kept**, because a real car camera sees the same sun, night and distance. Brightness and contrast augmentation was used instead.

### 4. Feature Selection
- Size columns (Width, Height, sign width, file size) are correlated 0.93-1.00, and the colour columns and brightness are correlated 0.92-1.00.
- The sign class explains only 8-30% of the variation in these simple features, so at least 70% of the differences between images have nothing to do with the sign type.
- The CNNs use the raw **pixels** instead: 64 x 64 input for the custom CNN, 128 x 128 for the pretrained networks, colour (RGB) kept, the whole image used (not the ROI box), and each channel normalised with the ImageNet mean and standard deviation.

### 5. Train / Validation / Test Split
- **Test:** the official 12,630 test images, kept apart until the end.
- **Validation:** about 10% of the training **tracks** of every class (at least one track per class), giving 35,219 training and 3,990 validation images with no track in both.
- A Random Forest baseline proved why this matters: **69.2%** accuracy with a random 5-fold split, but only **15.7%** when the folds were split by track (GroupKFold).

### 6. Deep Learning Model Development

This project uses supervised multi-class image classification to predict the sign type.

The following models were implemented and evaluated:

1. Random Forest on simple image features (baseline)
2. Custom CNN built from scratch (4 convolution blocks, 64 x 64)
3. MobileNetV3 as a frozen feature extractor, with a grid search on learning rate and weight decay
4. MobileNetV3-Large fine-tuned
5. EfficientNet-B0 fine-tuned

Training used data augmentation on the GPU (rotation, zoom, shift, brightness and contrast), with **no left-right flip** because a flipped "Turn left ahead" becomes "Turn right ahead". The models were trained with cross-entropy loss with label smoothing 0.1, the AdamW optimizer, a one-cycle learning rate schedule and mixed precision.

### 7. Model Evaluation and Comparison

The classes are imbalanced (210 to 2,250 training images per class), so models were evaluated using:
- Test Accuracy
- Macro F1 (every class counts the same)
- Top-3 Accuracy
- Number of test errors

| Model | Test Accuracy | Macro F1 | Top-3 Accuracy | Test Errors | Speed (images/s) |
|---|---:|---:|---:|---:|---:|
| Custom CNN | **99.83%** | **0.998** | **99.98%** | **22** | **2,676** |
| EfficientNet-B0 (fine-tuned) | 99.60% | 0.995 | 99.95% | 50 | 1,388 |
| MobileNetV3 (fine-tuned) | 99.18% | 0.985 | 99.87% | 104 | 1,871 |
| MobileNetV3 (feature extractor) | 73.3% | 0.654 | 90.7% | 3,368 | - |
| Random Forest (simple features) | 16.4% | 0.116 | 35.0% | 10,554 | - |

### 8. Test-Time Augmentation and Ensembles
- **Test-time augmentation (TTA)** with 3 zoom views (original, zoomed in 10%, zoomed out 10%).
- **Soft Voting** (average of the three CNNs' probabilities).
- **Stacking** (Logistic Regression trained on the validation predictions of the three CNNs).

| Model | Test Accuracy | Macro F1 | Test Errors |
|---|---:|---:|---:|
| Custom CNN + TTA | 99.80% | 0.997 | 25 |
| EfficientNet-B0 + TTA | 99.60% | 0.995 | 51 |
| MobileNetV3 + TTA | 99.18% | 0.984 | 103 |
| Soft Voting (3 CNNs + TTA) | 99.68% | 0.996 | 41 |
| Stacking (3 CNNs + TTA) | 99.64% | 0.995 | 46 |

None of these beat the single custom CNN. The training augmentation already covers zoom, and averaging with weaker models adds their mistakes.

**Best Performing Model: Custom CNN**

The custom CNN achieved the highest test accuracy (99.83%) and macro F1 (0.998), with only 22 errors on 12,630 test images. It is also the smallest and fastest model (1.18 million parameters, 5.1 minutes of training), which matters for reading signs from a video stream on a small onboard computer.

## Key Findings

- Pixels beat hand-made features: the Random Forest on colour and size columns reached 16.4%, while every CNN reached above 99%.
- A small CNN trained from scratch at the images' own resolution beat the larger pretrained networks, because traffic signs are standard drawings.
- A frozen ImageNet network was not enough (73.3%). Fine-tuning added 25.9 points for MobileNetV3.
- The most frequent mistake was **Speed limit 60 read as 80** (5 images). Most errors were between look-alike signs of the same shape and colour.
- Every class had an F1 of 0.978 or more. Correct predictions had a median confidence of **0.909**, wrong ones only **0.333**.
- Dark images were not worse within their class (-0.08 points), but small, far-away images were the real weakness (0.49% vs 0.08% error).

## Recommendations for the Driver-Assistance Team

- Act only on confident predictions: at confidence 0.7 or more, the model answers for 97.3% of images and is right 99.992% of the time. The images below 0.7 contain 21 of the 22 errors.
- Combine several frames of a track: averaging the first 3 frames recognised 132 of 133 validation tracks (99.25%).
- Invest in a higher-resolution or zoomed camera rather than better night vision, since distance matters more than light.
- Collect more different physical signs for the weak small classes, such as Bumpy road, Go straight or left, Roundabout and Dangerous curve left.
- Require agreement over several frames for speed limits and check them against map/GPS speed-limit data.
- Review suspicious labels, and add data from other countries, rain, snow and night before using the model outside these conditions.

## Challenges and Solutions

Some of the main challenges encountered included:

- **No sign names in the data:** Added the official GTSRB names and checked them against the shape and colour codes.
- **Data leakage between video frames:** Split validation by track and measured the effect with KFold vs GroupKFold (69.2% vs 15.7%).
- **Classes with only 7 tracks:** Picked about 10% of the tracks of every class, at least one per class, so all classes appear in validation.
- **Class imbalance:** Used a class-stratified split, macro F1 next to accuracy, and compared small and large classes inside each sign family.
- **Small, dark and blurry images:** Resized to 64 x 64, kept outliers, and used brightness, contrast, rotation, zoom and shift augmentation.
- **Left/right meaning of signs:** Used no horizontal flip in training or TTA.
- **Label smoothing caps confidence at about 0.9:** Chose confidence thresholds of 0.7-0.85 instead of 0.95.

## Disclaimer

This project was completed for educational and portfolio purposes as part of the DataMites AI Engineer Program. The model was trained and tested only on German traffic sign images and should not be interpreted as a validated production driver-assistance system.

## Author

**Suchitra Thanumalayan**

Data Science | Machine Learning | Artificial Intelligence
