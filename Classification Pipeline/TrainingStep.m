%%% ################################################################### %%%
%%% This function is the training cost function that optimizes the number  
%%% of spatial reference GLCM texture features matrices from each class using 
%%% Bayesian optimization and polynomial SVM classifier.
%%%
%%%                      INPUTS:
%%%
%%% z              : optimizated variable (spatial representative features)
%%% I_C_mean       : Ordered samples of Class 1 (COVID) based on Spearman's rank 
%%% I_NC_mean      : Ordered samples of Class 2 (Non-COVID) based on Spearman's rank
%%% Distance       : Wasserstein distance matrix for the training set
%%% N_patients_C   : Number of patients (samples) from class 1 (COVID)
%%% N_patients_NC  : Number of patients (samples) from class 2 (Non-COVID)
%%% outcome1       : Outcome of the training set 
%%% ord            : Order of the SVM's polynomial kernel
%%%
%%%                     OUTPUTS:
%%% 
%%% fun :  Cost function to minimize (1-accuracy) <--> maximize (accuracy) s.t. accuracy <=1
%%% 
%%% 
%%% 
%%% Written by Z. Belkhatir, 4/02/2021
%%% ################################################################### %%%
%%
function [fun] = TrainingStep(z,I_C_mean,I_NC_mean,Distance,N_patients_C,N_patients_NC,featC2,outcome1,ord)


rng('default')
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
        for i=1:length(sample_enhenced)  
            
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

c = cvpartition(table2array(trainingResponse),'KFold',5);
classificationSVM = fitcsvm(trainingPredictors,trainingResponse,'KernelFunction','polynomial','PolynomialOrder',ord,...
                    'BoxConstraint',1,'Standardize', true,'ClassNames', [0;1],'KernelScale','auto');
                
% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationSVM = classificationSVM;

 partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'CVPartition',c);
% 
% % Compute validation predictions
 [validationPredictions, validationScores] = kfoldPredict(partitionedModel);


% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
T = confusionmat(table2array(response),validationPredictions,'order',[0,1]);


[CMM,sensitivity,specificity,precision,accuracy,F1_score] = statistics(T);

fun = (1 - accuracy);

end