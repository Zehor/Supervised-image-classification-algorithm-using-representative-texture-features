%%% ################################################################### %%%
%%% This function is to test the performance of the optimized trained 
%%% classifier using test dataset, and the optimized reference samples.
%%% 
%%%                      INPUTS:
%%%
%%% z                 : Optimized number of reference samples
%%% I_C_mean          : Ordered training samples of Class 1 (COVID) based on Spearman's rank
%%% I_NC_mean         : Ordered training samples of Class 2 (Non-COVID) based on Spearman's rank
%%% Distance          : Wasserstein distance matrix for the training set
%%% N_patients_C      : Number of patients (samples) from class 1 (COVID)
%%% N_patients_NC     : Number of patients (samples) from class 2 (Non-COVID)
%%% featC2            : Statistical (scalar) features from GLCM of training set
%%% outcome1          : Outcome of the training set 
%%% ord               : Order of the SVM's polynomial kernel
%%% featC2_test       : Statistical (scalar) features from GLCM of test set
%%% outcome_test      : Outcome of the test set
%%% Distance_test     : Wasserstein distance matrix for the test set
%%%
%%%                     OUTPUTS: 
%%% 
%%% classificationSVM : Optimal trained SVM classifier
%%% T                 : COnfusion matrix of the validation prediction
%%% AUC               : Area under the curve for testing performance
%%% ROC_X             : X coordinate of the ROC curve
%%% ROC_Y             : Y coordinate of the ROC curve 
%%%
%%%
%%% Written by Z. Belkhatir, 4/02/2021
%%% ################################################################### %%%
%%
function [classificationSVM,T,AUC,ROC_X,ROC_Y] = ValidationStep(z,I_C_mean,I_NC_mean,Distance,N_patients_C,N_patients_NC,featC2,outcome1,ord,featC2_test,outcome_test,Distance_test)

sample_enhenced = [I_C_mean(1:z.n1),I_C_mean(end-z.n2+1:end)];
sample_Nenhenced =[I_NC_mean(1:z.n1),I_NC_mean(end-z.n2+1:end)];
 
reference= [sample_enhenced,sample_Nenhenced];
ind = 1:(N_patients_C + N_patients_NC);
ind = setdiff(ind,reference);
if (z.n1 == 0 & z.n2 ==0)
featAll=[];

for i=1:length(ind)

     features_all(i,:) = [cell2mat(featC2{ind(i),1})]'; 
end
else
  for j=1:length(ind)
          
        for i=1:length(sample_enhenced)  % Compute distance between each test data and all the training data 
            
            distance_enhenced_W1_all(j,i) = Distance(ind(j),sample_enhenced(i));
            distance_Nenhenced_W1_all(j,i) = Distance(ind(j),sample_Nenhenced(i));
            
        end
        featAll{j,1} = [distance_enhenced_W1_all(j,:),distance_Nenhenced_W1_all(j,:)]';
  end 

for i=1:length(ind)

     features_all(i,:) = [cell2mat(featC2{ind(i),1});featAll{i,1}]'; 
end
end
FeatAU(:,:) = features_all;


Feat = [mean(FeatAU,3),outcome1(ind)];

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

% Set up holdout validation
trainingPredictors = predictors;
trainingResponse = response;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.

classificationSVM = fitcsvm(trainingPredictors,trainingResponse,'KernelFunction','polynomial','PolynomialOrder',ord,...
   'Standardize', true,'ClassNames',[0;1],'BoxConstraint',1,'KernelScale','auto');


svmtrainFcn = @(x) predict(classificationSVM, x);
trainingPredictFcn = @(x) svmtrainFcn(x);
% 
% % Create the result struct with predict function
 svmPredictFcn = svmtrainFcn;
 validationPredictFcn = trainingPredictFcn;

% Add additional fields to the result struct
[trainingPredictions, trainingScores] = trainingPredictFcn(trainingPredictors);
% Compute validation accuracy
correctPredictions = (trainingPredictions == table2array(trainingResponse));
isMissing = isnan(table2array(trainingResponse));
correctPredictions = correctPredictions(~isMissing);
trainingAccuracy = sum(correctPredictions)/length(correctPredictions);
T1 = confusionmat(table2array(trainingResponse),trainingPredictions,'order',[0,1]);
%%
if (z.n1 == 0 & z.n2 ==0)
featAll_test=[];
for i=1:length(featC2_test)
     features_all_test(i,:) = [cell2mat(featC2_test{i,1})]'; 
end
else
 for j=1:length(featC2_test)
          
        for i=1:length(sample_enhenced)  % Distance between each test data and all the training data 
            
            distance_enhenced_W1_all(j,i) = Distance_test(j,sample_enhenced(i));
            distance_Nenhenced_W1_all(j,i) = Distance_test(j,sample_Nenhenced(i));
            
        end
        featAll_test{j,1} = [distance_enhenced_W1_all(j,:),distance_Nenhenced_W1_all(j,:)]';
  end 

for i=1:length(featC2_test)

     features_all_test(i,:) = [cell2mat(featC2_test{i,1});featAll_test{i,1}]'; 
end
end
FeatAU_test(:,:) = features_all_test;


Feat = [FeatAU_test,outcome_test];

inputTable = array2table(Feat); 
[n,m] = size(inputTable);
for i=1:m-1
    name = strcat('Feat',num2str(i));
    predictorNames{1,i} = name;
end
%

validationPredictors = inputTable(:, predictorNames);
name_res = strcat('Feat',num2str(m));
validationResponse= inputTable(:,name_res);


% Compute validation predictions

[validationPredictions, validationScores] = trainingPredictFcn(validationPredictors);
 %plotconfusion(table2array(validationResponse),validationPredictions)
[ROC_X,ROC_Y,~,AUC] = perfcurve(table2array(validationResponse),validationScores(:,1),0);
% % Compute validation accuracy
% % correctPredictions = (validationPredictions == table2array(validationResponse));
% % isMissing = isnan(table2array(validationResponse));
% % correctPredictions = correctPredictions(~isMissing);
% % validationAccuracy = sum(correctPredictions)/length(correctPredictions);
T = confusionmat(table2array(validationResponse),validationPredictions,'order',[0,1]);

end
