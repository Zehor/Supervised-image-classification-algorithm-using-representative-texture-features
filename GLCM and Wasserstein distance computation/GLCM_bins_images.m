%%% ################################################################### %%%
% This function computes the co-occurance matrix using CERR package 
%
%                                  INPUTS
%
% dirName     : Path to directory with CT images.
% numGrLevels : No. gray levels
% dirctn      : 1- Returns directions of 13 neighbours in 3D (CERR/DICOM files), 
%               2- Returns directions of 4 neightbours (2D)
% cooccurType : 1- returns a single cooccurrence matrix by combining
%               contributions from all offsets into one cooccurrence
%               matrix.
%               2- returns cooccurM with each column containing
%               cooccurrence matrix for the row of offsetsM.
%
%                                OUTPUTS
%
% cooccurC   : Spatial GLCM matrix of all images  
% featC      : Scalar features from the cooccurC

% Example:
% dirName = 'Path/to/CT_images file';
% numGrLevels = 32;
% dirctn      = 2; % 2d neighbors
% cooccurType = 1; % Combining contributions from all offsets into one cooccurrence matrix.
% [cooccurC,featC] = batchGetGLCM(dirName,numGrLevels,dirctn,cooccurType);
%------------------------------------------------------------------------
%Written by Z. Belkhatir, 4/02/2021

function [cooccurC,featC] = GLCM_bins_images(dirName,numGrLevels,dirctn,cooccurType)

%Set flags for required features
glcmFlagS.energy = 1;
glcmFlagS.jointEntropy = 1;
glcmFlagS.jointMax = 1;
glcmFlagS.jointAvg = 1;
glcmFlagS.jointVar = 1;
glcmFlagS.contrast = 1;
glcmFlagS.invDiffMoment = 1;
glcmFlagS.sumAvg = 1;
glcmFlagS.corr = 1;
glcmFlagS.clustShade = 1;
glcmFlagS.clustProm = 1;
glcmFlagS.haralickCorr = 1;
glcmFlagS.invDiffMomNorm = 1;
glcmFlagS.invDiff = 1;
glcmFlagS.invDiffNorm = 1;
glcmFlagS.invVar = 1;
glcmFlagS.dissimilarity = 1;
glcmFlagS.diffEntropy = 1;
glcmFlagS.diffVar = 1;
glcmFlagS.diffAvg = 0; %SAME AS DISSIMILARITY
glcmFlagS.sumVar = 1;
glcmFlagS.sumEntropy = 1;
glcmFlagS.clustTendency = 1;
glcmFlagS.autoCorr = 1;
glcmFlagS.invDiffMomNorm = 1;
glcmFlagS.firstInfCorr = 1;
glcmFlagS.secondInfCorr = 1;

% Iterate over all CT images in the directory
dirS1 = dir([dirName,filesep,'*.png']);
dirS2 = dir([dirName,filesep,'*.jpg']);
nameC = {dirS1.name,dirS2.name};
cooccurC = cell(length(nameC),1);
featC = cell(length(nameC),1);


for planNum = 1:length(nameC)
        quantized3M = imread(nameC{planNum});
        quantized3M = rgb2gray(quantized3M);
        % Build cooccurrence matrix
        offsetsM = getOffsets(dirctn);
        cooccurM = calcCooccur(quantized3M, offsetsM, numGrLevels, cooccurType);
        
        cooccurC{planNum, str} = reshape(cooccurM,numGrLevels,numGrLevels);
        
        featureS = cooccurToScalarFeatures(cooccurM, glcmFlagS);
        featureS = rmfield(featureS,'diffAvg'); %SAME AS DISSIMILARITY
        featC{planNum, str} = struct2cell(featureS);
    
end

end