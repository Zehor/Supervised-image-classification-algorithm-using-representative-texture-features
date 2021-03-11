%%% ################################################################### %%%
%%% This script compute the Wassesrtein-1 distance between any two 2-D images 
%%% It is applied to compute W1 between all pairs of GLCMs in the cohort.
%%% It uses a fast numerical method and algorithm that was proposed by 
%%% J. Liu et al. in the following paper:
%%% " Multilevel Optimal Transport: A fast approximation of Wasserstein-1 Distances, 2018" 
%%%
%%% Written by Z. Belkhatir, 4/02/2021
%%% ################################################################### %%%


clear all 
close all
clc
%%
currentpath = pwd;
addpath([currentpath, '/Data']);

load('GLCM_COVID_all_dir_32_Seg.mat')

N_patients_C = 174;
N_patients_NC = 150;

for i=1:(N_patients_C+N_patients_NC)
    for j=(i+1):(N_patients_C+N_patients_NC)

        img1= cell2mat(cooccurC(i,1));
        img2= cell2mat(cooccurC(j,1));

        Distance (i,j) = W1_images(img1,img2,2);
        Distance (j,i) = Distance(i,j); 
    end
end
