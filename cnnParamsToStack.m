function [Wh, bh, Ws, bs] = cnnParamsToStack(theta,cnn,numClasses)
% Converts unrolled parameters for a single layer convolutional neural
% network followed by a softmax layer into structured weight
% tensors/matrices and corresponding biases
%                            
% Parameters:
%  theta      -  unrolled parameter vectore
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  Wh      -  filterDim x filterDim x numFilters parameter matrix
%  Ws      -  numClasses x hiddenSize parameter matrix, hiddenSize is
%             calculated as numFilters*((imageDim-filterDim+1)/poolDim)^2 
%  bh      -  bias for convolution layer of size numFilters x 1
%  bs      -  bias for dense layer of size hiddenSize x 1

n = numel(cnn.layers);
Wh = cell(n-1);
indS = 0;
indE = 0;
inputMaps = cnn.layers{1}.channels;
for i = 1:n-1
    filterDim = cnn.layers{i+1}.filterDim;
    numFilters = cnn.layers{i+1}.numFilters;
    indS = indE + 1;
    indE = indE + filterDim^2*numFilters*inputMaps;
    Wh{i} = reshape(theta(indS:indE),filterDim,filterDim,inputMaps,numFilters);
    indS = indE + 1;
    indE = indE + numFilters;
    bh{i} = reshape(theta(indS:indE),numFilters,1);
    inputMaps = numFilters;
end

indS = indE +1;
indE = indE + cnn.fvSize*numClasses;
Ws = reshape(theta(indS:indE),numClasses,cnn.fvSize);
bs = theta(indE+1:end);



end