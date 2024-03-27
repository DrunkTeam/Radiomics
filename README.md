# Radiomics
Detection k27m mutation in glioms of the brain stem.

## Team
- Ninel Yunusova
- Sofia Polozova
- Dmitrij Kamyshnikov

## Goal
The purpose of our project is to classify the mutational tumor k27m of the brain stem. Glioblastoma (GBM) is the most common primary malignant tumor of the central nervous system with characteristics of highly aggressive growth, high recurrence rate and poor prognosis. Diffuse midline glioma (DMG), k27m‐mutant is a new nosology introduced in the WHO classification of CNS tumors in 2016, which is extremely rare (our sample n = 200). A typical localization is the midline of the brain, which includes the thalamus, hypothalamus, pineal gland, bridge and spinal cord. As input, we receive a series of MRI images with various tumors, weighted by T2. A mask with the location of the tumor is attached to the series. As part of the project, we plan to experiment with various classification methods to more accurately determine the mutation in the images. In our case, it is preferable to focus on the recall metric as a quality metric, since it is important for us not to miss a single mutation case, and also to take into account accuracy. Based on the articles studied, our main task will be to increase the accuracy of mutation prediction to 90%.

## Data
We have 200 series of t2-weighted MRI images with maskk of tumor. 50 of them have the k27m mutation. T2-weighted series presented .nrrd files.

### Preparation of the data
For retrieve readiomics feature we use [pyRadiomics](https://www.radiomics.io/pyradiomics.html), which works with NIFTI series. 
Using [Slicer3D](https://www.slicer.org/) we checked series and convert .nrrd to .nifti.

