function [theta,fvSize] = cnnInitParams(cnn,numClasses)
% Initialize parameters for multiple layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  cnn      -    net structure
%  numClasses -  number of classes to predict
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% Initialize parameters randomly based on layer sizes.

%params:
%   Wh   -  hidden layer weight
%   bh   - hidden layer bais
%   Ws   - softamx layer weight
%   bs   - somtmax layer bais

n = numel(cnn.layers);
Wh = cell(n-1);
for i = 1:n
    if(strcmp(cnn.layers{i}.type,'i'))
        inputDim = cnn.layers{i}.imageDim;
        inputMaps = cnn.layers{i}.channels;
    elseif(strcmp(cnn.layers{i}.type,'h'))
        filterDim = cnn.layers{i}.filterDim;
        poolDim = cnn.layers{i}.poolDim;
        outputMaps = cnn.layers{i}.numFilters;
        Wh{i-1} = 1e-1*randn(filterDim,filterDim,inputMaps,outputMaps);
        bh{i-1} = zeros(outputMaps,1);
        
        %update for next layer
        outDim = (inputDim - filterDim + 1)/poolDim;
        inputDim = outDim;
        inputMaps = outputMaps;
    end
end

featureSize = outDim^2*outputMaps;
fvSize = featureSize;

% we'll choose weights uniformly from the interval [-r, r]
r  = sqrt(6) / sqrt(numClasses+featureSize+1);
Ws = rand(numClasses,featureSize) * 2 * r - r;

bs = zeros(numClasses, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc.
theta = [];
for i = 1:n-1
    theta = [theta; Wh{i}(:); bh{i}(:)];
end

theta = [theta; Ws(:); bs(:)];


end