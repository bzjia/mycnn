function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                               cnn,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
imageCh = size(images,3);
numImages = size(images,4); % number of images
lambda = 3e-3; % weight decay parameter     

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
n = numel(cnn.layers);

[Wh, bh, Ws, bs] = cnnParamsToStack(theta,cnn,numClasses); %the theta vector cosists wc,wd,bc,bd in order

% Same sizes as Wh,Ws,bh,bs. Used to hold gradient w.r.t above params.

%Ws_grad = zeros(size(Ws));
%bs_grad = zeros(size(bs));
%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
%fprintf('forward propagation...\n');
for i = 1:n

    if(strcmp(cnn.layers{i}.type,'i'))
        inputDim = cnn.layers{i}.imageDim;
        inputMaps = cnn.layers{i}.channels;
        inputFeatures = images;
    elseif(strcmp(cnn.layers{i}.type,'h'))
        filterDim = cnn.layers{i}.filterDim;
        poolDim = cnn.layers{i}.poolDim;
        outputMaps = cnn.layers{i}.numFilters;
        

        convDim = inputDim-filterDim+1; % dimension of convolved output
        outputDim = (convDim)/poolDim; % dimension of subsampled output

        convolvedFeatures = cnnConvolve(filterDim,inputMaps,outputMaps,inputFeatures,Wh{i-1},bh{i-1});
        %forward pool
       % convolvedFeatures = sigm(convolvedFeatures);
        pooledFeatures = cnnPool(poolDim, convolvedFeatures);
        %store in cnn
        cnn.layers{i}.convolvedFeatures = convolvedFeatures;
        cnn.layers{i}.pooledFeatures = pooledFeatures;
        cnn.layers{i}.outputDim = outputDim;
        %update for next layer
        inputFeatures = pooledFeatures;
        inputMaps = outputMaps;
        inputDim = outputDim;
    end

end
% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(pooledFeatures,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
%probs的每一列代表一个输出
Ma = Ws*activationsPooled+repmat(bs,[1,numImages]); 
NorM = bsxfun(@minus, Ma, max(Ma, [], 1));  %归一化，每列减去此列的最大值，使得M的每个元素不至于太大。
ExpM = exp(NorM);
probs = bsxfun(@rdivide,ExpM,sum(ExpM));      %概率


%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
% cost = -1/numImages*labels(:)'*log(probs(:));
% 首先需要把labels弄成one-hot编码
groundTruth = full(sparse(labels, 1:numImages, 1));
cost = -1./numImages*groundTruth(:)'*log(probs(:))+(lambda/2.)*(sum(Ws(:).^2)); %加入一个惩罚项
% cost = -1./numImages*groundTruth(:)'*log(probs(:));
%fprintf('cost:%f\n',cost);
% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%======================================================================
%  Backpropagation


%softmax layer grad
%fprintf('back propagation...\n');
delta_d = -(groundTruth-probs); % softmax layer's preactivation,每一个样本都对应有自己每层的误差敏感性。
Ws_grad = (1./numImages)*delta_d*activationsPooled'+lambda*Ws;
bs_grad = (1./numImages)*sum(delta_d,2); %注意这里是要求和
n = numel(cnn.layers)+1;
delta = cell(n);
delta{n} = Ws'*delta_d;
outputDim = cnn.layers{n-1}.outputDim;
numFilters = cnn.layers{n-1}.numFilters;
delta{n} = reshape(delta{n},outputDim,outputDim,numFilters,numImages);
outputMaps = cnn.layers{n-1}.numFilters;
poolDim = cnn.layers{n-1}.poolDim;

for j=1:outputMaps
   a = cnn.layers{n-1}.convolvedFeatures;
   delta{n-1}(:,:,j,:) = squeeze(a(:,:,j,:).*(1-a(:,:,j,:))).*expand(squeeze(delta{n}(:,:,j,:)),[poolDim,poolDim,1])/poolDim^2;
end


for l = n-2:-1:2
    %pool
    inputMaps = cnn.layers{l}.numFilters;
    outputMaps = cnn.layers{l+1}.numFilters;
    delta_tmp = zeros(size(cnn.layers{l}.pooledFeatures));
    
    for i=1:inputMaps
       z = zeros(size(cnn.layers{l}.pooledFeatures(:,:,1,:)));
       for j = 1:outputMaps
           z = z + convn(delta{l+1}(:,:,j,:),rot180(Wh{l}(:,:,i,j)),'full');
       end
       delta_tmp(:,:,i,:) = z;
    end
    
    %convolution
    poolDim = cnn.layers{l}.poolDim;
     
    for j=1:cnn.layers{l}.numFilters
        a = cnn.layers{l}.convolvedFeatures;
        delta{l}(:,:,j,:) = squeeze(a(:,:,j,:).*(1-a(:,:,j,:))).*expand(squeeze(delta_tmp(:,:,j,:)),[poolDim,poolDim,1])/poolDim^2;
    end
    
end

% Gradient Calculation

Wh_grad = cell(n-2);
bh_grad = cell(n-2);
for l = 1:n-2
    Wh_grad{l} = zeros(size(Wh{l}));
    bh_grad{l} = zeros(size(bh{l}));
end
inputFeature = images;
for l = 2:n-1
    for j = 1:cnn.layers{l}.numFilters
        for i = 1:cnn.layers{l-1}.numFilters    
           Wh_grad{l-1}(:,:,i,j) = (1./numImages)*convn(flipall(inputFeature(:,:,i,:)),delta{l}(:,:,j,:),'valid')+lambda*Wh{l-1}(:,:,i,j); 
        end
        dj = delta{l}(:,:,j,:);
        bh_grad{l-1}(j) = ((1./numImages))*sum(dj(:));
    end
    inputFeature = cnn.layers{l}.pooledFeatures;
end

%% Unroll gradient into grad vector for minFunc
grad = [];
for l = 1:n-2
    grad = [grad;Wh_grad{l}(:);bh_grad{l}(:)];
end
grad = [grad;Ws_grad(:);bs_grad(:)];
end

function X = rot180(X)
    X = flipdim(flipdim(X, 1), 2);
end
function X = flipall(X)
    for i =1:ndims(X)
        X = flipdim(X,i);
    end
end