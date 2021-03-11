clear all
close all
clc
%%
currentpath = pwd;
addpath([currentpath, '/Data']);

% Number of samples in each class 
N_patients_C = 174;
N_patients_NC = 150;

% Binary outcome (classification)
outcome1 = [zeros(N_patients_NC,1);ones(N_patients_C,1)];
Trials = 1; 


% Wasserstein distance between all pairs of the images' GLCMS
load('Distance_GLCM_all_NC_C_dir_32_Seg.mat')

% 2D GLCMs and Statistical features extracted them (see list of features in
% List_of_features.txt)
load('GLCM_COVID_all_dir_32_Seg.mat')

% Order of the SVM polynomial kernel
ord = 3;

% Binary outcome (classification)
outcome = [zeros(N_patients_NC,1);ones(N_patients_C,1)];

% Run traning/validation phases of the proposed classification algorithm
[classification_statistics_test] = Classification_pipeline_main(N_patients_C,N_patients_NC,Distance,...
                                                                featC,Trials,outcome,ord);