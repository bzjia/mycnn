%% Convolution Neural Network Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.

% Configuration

cnn.layers = {
    struct('type','i','imageDim',180,'channels',3,'numFilters',3);
    struct('type','h','numFilters',3,'filterDim',13,'poolDim',7);
    %struct('type','h','numFilters',16,'filterDim',5,'poolDim',1);
    };


% Load  Train data

% load stlTrainSubset.mat;
% %images = trainImages(:,:,:,:);
% images = zeros(size(trainImages,1),size(trainImages,2),1,size(trainImages,4));
% for i = 1:numTrainImages
%     images(:,:,1,i) = rgb2gray(trainImages(:,:,:,i));
% end
% labels = trainLabels(:,1);
load traindata_sample.mat;
labels = labels+1;
% imageDim = cnn.layers{1}.imageDim;
% inputCh = 1;
% images = loadMNISTImages('./mnist/train-images-idx3-ubyte');
% images = reshape(images,imageDim,imageDim,inputCh,[]);
% labels = loadMNISTLabels('./mnist/train-labels-idx1-ubyte');
% labels(labels==0) = 10; % Remap 0 to 10

% Initialize Parameters
numClasses = 2;
[theta,fvSize] = cnnInitParams(cnn,numClasses);
cnn.fvSize = fvSize;

%%======================================================================
%% STEP 1: Implement convNet Objective
%  Implement the function cnnCost.m.

%%======================================================================
%% STEP 2: Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

DEBUG=true;;  % set this to true to check gradient
%DEBUG = true;
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
            
            
    db_images = images1(:,:,:,1:10);
    db_labels = labels(1:10,1);
   db_theta = theta;
    
    [cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,cnn);
    save('grad.mat','grad');

    % Check gradients
    numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
                                db_labels,numClasses,cnn), db_theta,grad);
 
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff); 
 
    assert(diff < 1e-9,...
        'Difference too large. Check your gradient computation again');
    
end;

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

% 因为是采用的mini-batch梯度下降法，所以总共对样本的循环次数次数比标准梯度下降法要少
% 很多，因为每次循环中权值已经迭代多次了
mini_batch = 1;
if (mini_batch == false)
    addpath minFunc/
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                              % function. Generally, for minFunc to work, you
                              % need a function pointer with two outputs: the
                              % function value and the gradient. In our problem,
                              % sparseAutoencoderCost.m satisfies this.
    options.maxIter = 10;      % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';


    [opttheta,~] = minFunc(@(p)cnnCost(p,images,labels,numClasses,cnn),theta,options)
else
    options.epochs = 3; 
    options.minibatch = 256;
    options.alpha = 1e-1;
    options.momentum = .95;

    opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,cnn),theta,images,labels,options);
end
save('theta.mat','opttheta');             

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

load stlTestSubset.mat;
%testImages = testImages(:,:,:,:);
Images = zeros(size(testImages,1),size(testImages,2),1,size(testImages,4));
for i = 1:numTrainImages
    Images(:,:,1,i) = rgb2gray(testImages(:,:,:,i));
end
testLabels = testLabels(:,1);

% testImages = loadMNISTImages('./mnist/t10k-images-idx3-ubyte');
% testImages = reshape(testImages,imageDim,imageDim,inputCh,[]);
% testLabels = loadMNISTLabels('./mnist/t10k-labels-idx1-ubyte');
% testLabels(testLabels==0) = 10; % Remap 0 to 10

[Y,cost,preds]=cnnCost(opttheta,Images,testLabels,numClasses,...
                cnn,true);

acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);