function [MCC,sensitivity,specificity,precision,accuracy,F1_score] = statistics(T)
% Compute classification performance metrics from the confusion matrix
%------------------------------------------------------------------------
% INPUTS
%
% T          : Confusion matrix
%
% OUTPUTS
%
% MCC        : Mathew Correlation coefficient
% sensitivity: sensitivity 
% specificity: specificity
% precision  : precision
% accuracy   : accuracy
% F1_score   : F1 score
%------------------------------------------------------------------------
%Written by Z. Belkhatir, 4/02/2021

TP = T(1,1);
FP = T(2,1);
TN = T(2,2);
FN = T(1,2);

MCC = (TP .* TN - FP .* FN) ./ ...
    sqrt( (TP + FP) .* (TP + FN) .* (TN + FP) .* (TN + FN) );

sensitivity = TP /(TP+FN);

specificity = TN / (TN+FP);

precision = TP / (TP+FP);

accuracy = (TP+TN) / (TP+TN+FP+FN);

F1_score = (2*TP) /(2*TP + FN +FP);

end

