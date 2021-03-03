clear all
close all
clc
%%

% Number of samples in each class 
N_patients_C = 174;
N_patients_NC = 150;

% Binary outcome (classification)
outcome1 = [zeros(N_patients_NC,1);ones(N_patients_C,1)];
Trials = 1; 


% Wasserstein distance between all pairs of the images' GLCMS
load('../Data/Distance_GLCM_all_NC_C_dir_all_32_Seg.mat')

% Statistical features extracted from 2D GLCMs (see list of features in
% List_of_features.txt)
load('../Data/GLCM_COVID_all_dir_all_32_Seg.mat')

% Order of the SVM polynomial kernel
ord = 3;

% Binary outcome (classification)
outcome = [zeros(N_patients_NC,1);ones(N_patients_C,1)];

% Run traning/validation phases of the proposed classification algorithm
[classification_statistics_test] = Classification_pipeline_main(N_patients_C,N_patients_NC,Distance,...
                                                                featC,Trials,outcome,ord);