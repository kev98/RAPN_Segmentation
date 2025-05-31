# Surgical Tools and Organs Segmentation in Robot-Assisted Procedures
This repository contains the code I developed for my Master's thesis during an internship program completed at the Orsi Academy (Belgium). The work focuses on developing neural network models for surgical scene segmentation. The first step involved working with a team of doctors and engineers to create the largest dataset of laparoscopic images of partial nephrectomy procedures, complete with annotated masks for surgical tool segmentation (private nowadays). The project covers the numerous steps taken to create this dataset, including the statistical analyses that were carried out. Then, through in-depth analysis of benchmark datasets, the best state-of-the-art neural networks were selected and trained on the created dataset to
obtain good binary and multiclass segmentation models of surgical instruments on our internal dataset.

<figure>
 <img style="float: left" src="figs/Segmentation_results.png" alt="Side view" width="100%">
 <figcaption><em>Some masks predicted by the multiclass model (20 classes) using DeepLabV3+ with EfficientNet-B4.</em></figcaption>
</figure>

In a second step, the project also explores how temporal consistency can be leveraged for the segmentation of surgical instruments in videos. For the annotations of the videos we used a pseudolabeling technqieus
