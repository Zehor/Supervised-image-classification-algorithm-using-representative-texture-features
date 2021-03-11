# Supervised Image Classification Algorithm Using Representative Spatial Texture Features

This MATLAB code is useful to reproduce results of the paper "Supervised Image Classification Algorithm Using Representative Spatial Texture Features: Application
to COVID-19 Diagnosis Using CT Images" submitted to the Journal of Machine Learning Research. An medrxiv version of the paper can be found in https://www.medrxiv.org/content/10.1101/2020.12.03.20243493v1.full.pdf

To run the code properly, please make sure that you install MATLAB or have access to the online version.

# Copyright (c) Zehor Belkhatir, Raúl San José Estépar, and Allen R. Tannenbaum

# Contact Persons: Zehor Belkhatir 
# (E-mail: zohor.belkhatir@gmail.com)

* The MATLAB codes available in the folder "GLCM and Wasserstein distance compute" can be used for any set of images that you would like analyse using the    
  proposed texture feature approach.
    * GLCM_bins_images.m : Computes the 2D GLCM matrices and extracts the statistical features from them.
    * To compute the GLCM you need to add the "Computational Environment for Radiological Research (CERR)" MATLAB package to the search Path
    * Wasserstein_main.m : Computes the W1 distance between all pairs of GLCMs in the cohort. The W1 distance is computed using a fast numerical algorithm     
      proposed in " J. Liu et al., Multilevel Optimal Transport: A fast approximation of Wasserstein-1 Distances, 2018"
* The codes available in the folder "Classification pipeline" runs the proposed texture classification pipepline using the GLCM texture and Wasserstein distance 
  matrix data that are available in the folder "Data". Excute the script "Main_run.m" to run the algorithm.  
