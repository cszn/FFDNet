% This is the training demo of FFDNet for denoising noisy grayscale images corrupted by
% AWGN.
%
% To run the code, you should install Matconvnet (http://www.vlfeat.org/matconvnet/) first.
%
% @article{zhang2018ffdnet,
%   title={FFDNet: Toward a Fast and Flexible Solution for {CNN} based Image Denoising},
%   author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
%   journal={IEEE Transactions on Image Processing},
%   volume={27}, 
%   number={9}, 
%   pages={4608-4622}, 
% }
%
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

%% ********************  Note  **********************************
% ** You should set the training images folders of "generatepatches.m" first. Then you can run "Demo_Train_FFDNet_gray.m" directly.
% **
% ** folders    = {'path_of_your_training_dataset'};% set this from "generatepatches.m" first!
% ** stride     = 10;                       % control the number of image patches, from "generatepatches.m"
% ** nimages  = round(length(filepaths));   % control the number of image patches, from "generatepatches.m"
% **
%% **************************************************************




format compact;
addpath('utilities');

%-------------------------------------------------------------------------
% Configuration
%-------------------------------------------------------------------------

opts.modelName        = 'FFDNet_gray'; % model name
opts.learningRate     = [logspace(-4,-4,100),logspace(-4,-4,100)/3,logspace(-4,-4,100)/(3^2),logspace(-4,-4,100)/(3^3),logspace(-4,-4,100)/(3^4)];% you can change the learning rate
opts.batchSize        = 64; % default  
opts.gpus             = [1]; % this code can only support one GPU!
opts.numSubBatches    = 2;
opts.weightDecay      = 0.0005;
opts.expDir           = fullfile('data', opts.modelName);

%-------------------------------------------------------------------------
%  Initialize model
%-------------------------------------------------------------------------

net  = feval(['model_init_',opts.modelName]);

%-------------------------------------------------------------------------
%   Train
%-------------------------------------------------------------------------

[net, info] = model_train(net,  ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'numSubBatches',opts.numSubBatches, ...
    'weightDecay',opts.weightDecay, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;





