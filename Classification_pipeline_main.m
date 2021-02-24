%%% ################################################################### %%%
%%% This function runs the classification pipeline proposed in Figure 1 
%%% of the following paper: 
%%% "Supervised Image Classification Algorithm Using Representative Spatial 
%%% Texture Features: Application to COVID-19 Diagnosis Using CT Images",
%%% by Z. Belkhatir, R.S-J. Estepar, and A. R. Tannenbaum, 2020 
%%% Preprint available at: 
%%% https://www.medrxiv.org/content/10.1101/2020.12.03.20243493v1.full.pdf
%%%
%%%                      The inputs:
%%%
%%% N_patients_C  : Number of patients (samples) from class 1 (COVID)
%%% N_patients_NC : Number of patients (samples) from class 2 (Non-COVID)
%%% Distance      : Wasserstein distance matrix between all pairs of
%%%                 samples available (training + test)
%%% featC2        : statistical features from GLCMs of region of interest
%%%                 for all available sample (training + test)
%%% outcome       : Outcome (Class) for all available samples (training + test)
%%% ord           : Order of the SVM's polynomial kernel
%%%
%%%                     The outputs: 
%%% 
%%% classification_statistics_test : Classification metrics for all trials 
%%% 
%%% 
%%% 
%%% Written by Z. Belkhatir, 4/02/2021
%%% ################################################################### %%%
%%
function [classification_statistics_test] = Classification_pipeline_main(N_patients_C,N_patients_NC,Distance,featC2,Trials,outcome,ord)

rng('default');

for L=1:Trials 
    close all
    clc
    L 

indices_NC = 1:N_patients_NC;
indices_C  = N_patients_NC+1 : (N_patients_C + N_patients_NC) ;

ind = 1:(N_patients_C + N_patients_NC);


for i=1:length(ind)
    features_all(i,:) = [cell2mat(featC2{ind(i),1})]'; 
end
FeatAU = features_all;

Feat = [mean(FeatAU,3),outcome];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Split Data into training and test %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inputTable = array2table(Feat); 
[n,m] = size(inputTable);
for i=1:m-1
    name = strcat('Feat',num2str(i));
    predictorNames{1,i} = name;
end
%

predictors = inputTable(:, predictorNames);
name_res = strcat('Feat',num2str(m));
response = inputTable(:,name_res);

% Set up holdout validation 20% test and 80% training

cvp = cvpartition(outcome, 'Holdout', 0.2);


All = 1:cvp.NumObservations;
TrainingSet = All'.* cvp.training;
TrainingSet(TrainingSet==0) = [];

TestSet = All'.* cvp.test;
TestSet(TestSet==0) = [];

% Training and test data and output
Distance_test = Distance(TestSet,TrainingSet);
Distance_train = Distance(TrainingSet,TrainingSet);
N_patients_NC = length(intersect(TrainingSet,indices_NC));
N_patients_C = length(intersect(TrainingSet,indices_C));

outcome_training = outcome(TrainingSet);
outcome_test = outcome(TestSet);

featC_train = featC2(TrainingSet,:);
featC_test = featC2(TestSet,:);


% Rank the samples in each class based on defined fittness criteria

Distance_NC_NC = Distance_train(1:N_patients_NC,1:N_patients_NC);

Distance_C_C = Distance_train(N_patients_NC+1:end,N_patients_NC+1:end);

[RHO_NC_NC,~] = corr(Distance_NC_NC,'Type','Spearman');

[RHO_C_C,~] = corr(Distance_C_C,'Type','Spearman');

RHO_NC_NC = RHO_NC_NC- diag(diag(RHO_NC_NC));

RHO_C_C = RHO_C_C- diag(diag(RHO_C_C));

RHO_C_C_mean = mean(RHO_C_C);

RHO_NC_NC_mean = mean(RHO_NC_NC);


[~,I_NC_mean] = sort(RHO_NC_NC_mean,2,'descend');
[~,I_C_mean] = sort(RHO_C_C_mean,2,'descend');
I_C_mean = I_C_mean + N_patients_NC;


n1 = optimizableVariable('n1',[0,20],'Type','integer');
n2 = optimizableVariable('n2',[0,20],'Type','integer');


% Find optimal number of reference samples maximising the training accuracy

fun=@(z) TrainingStep(z,I_C_mean,I_NC_mean,Distance_train,N_patients_C,...
                      N_patients_NC,featC_train,outcome_training,ord);
                 
results = bayesopt(fun,[n1,n2],'MaxObjectiveEvaluations',300,...
           'IsObjectiveDeterministic',true,'AcquisitionFunctionName','expected-improvement-plus');

z_opt = results.XAtMinObjective; 

% Test the performance of the optimal trained SVM classifier on test data
[T,auc,roc_x,roc_y] = ValidationStep(z_opt,I_C_mean,I_NC_mean,Distance_train,...
              N_patients_C,N_patients_NC,featC_train,outcome_training,ord,featC_test,outcome_test,Distance_test);

% Compute statistics of the classification performance
con_ma(:,:,L) = T;
[MCC,sensitivity,specificity,precision,accuracy,F1_score] = statistics(T);
valid(L) = accuracy;
sen(L) = sensitivity;
spe(L) = specificity;
pre(L) = precision;
mcc(L) = MCC;
F1(L) = F1_score;
N1(:,L) = table2array(z_opt);
ROC_X(:,L) = roc_x;
ROC_Y(:,L) = roc_y;
AUC(L) = auc;

end
classification_statistics_test = {con_ma,valid,sen,spe,pre,mcc,F1,N1,ROC_X,ROC_Y,AUC};

end

