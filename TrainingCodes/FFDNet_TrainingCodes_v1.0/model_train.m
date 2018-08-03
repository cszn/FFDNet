function [net, state] = model_train(net, varargin)


% simple code 

%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option).

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%-------------------------------------------------------------------------
% solver:  Adam
%-------------------------------------------------------------------------

opts.beta1        = 0.9;
opts.beta2        = 0.999;
opts.alpha        = 0.01;
opts.epsilon      = 1e-8;
opts.weightDecay  = 0.0001;

%-------------------------------------------------------------------------
%  setting for simplenn
%-------------------------------------------------------------------------

opts.conserveMemory = true;
opts.mode           = 'normal';
opts.cudnn          = true ;
opts.backPropDepth  = +inf ;
opts.skipForward    = false;
opts.numSubBatches  = 1;

%-------------------------------------------------------------------------
%  setting for model
%-------------------------------------------------------------------------

opts.batchSize  = 128 ;
opts.gpus       = [];
opts.numEpochs  = 300 ;
opts.learningRate = 0.0001*ones(1,100,'single');
opts.modelName  = 'model';
opts.expDir     = fullfile('data',opts.modelName) ;

%-------------------------------------------------------------------------
%  update settings
%-------------------------------------------------------------------------

opts            = vl_argparse(opts, varargin);
opts.numEpochs  = numel(opts.learningRate);

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

%-------------------------------------------------------------------------
% Initialization
%-------------------------------------------------------------------------

net = vl_simplenn_tidy(net);    %%% fill in some eventually missing values
net.layers{end-1}.precious = 1;
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

state.getBatch = getBatch ;

%-------------------------------------------------------------------------
% Train and Test
%-------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'-epoch-%d.mat'], ep));

start = findLastCheckpoint(opts.expDir,opts.modelName) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d', mfilename, start) ;
    load(modelPath(start), 'net') ;
    % net = vl_simplenn_tidy(net) ;
end

imdb = [];

for epoch = start+1 : opts.numEpochs
    
    % Train for one epoch.
    state.epoch = epoch ;
    state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate)));
    if numel(opts.gpus) == 1
        net = vl_simplenn_move(net, 'gpu') ;
    end
    
    %-------------------------------------------------------------------------
    % generate training data
    %-------------------------------------------------------------------------
    
    if  mod(epoch,10)~=1 && isfield(imdb,'set') ~= 0
        
    else
        clear imdb;
        [imdb] = generatepatches;
    end
    opts.train = find(imdb.set==1);
    
    state.train  = opts.train(randperm(numel(opts.train))) ; % shuffle
    [net, state] = process_epoch(net, state, imdb, opts, 'train');
    %net.layers{end}.class =[];
    net          = vl_simplenn_move(net, 'cpu');
    % save current model
    save(modelPath(epoch), 'net')
    
end


function  [net, state] = process_epoch(net, state, imdb, opts, mode)
if strcmp(mode,'train')
    % solver: Adam
    for i = 1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            for j = 1:numel(net.layers{i}.weights)
                state.layers{i}.t{j} = 0;
                state.layers{i}.m{j} = 0;
                state.layers{i}.v{j} = 0;
            end
        end
    end
end

subset = state.(mode) ;
num = 0 ;
res = [];
for t=1:opts.batchSize:numel(subset)
    for s=1:opts.numSubBatches
        
        % get this image batch
        batchStart = t + (s-1);
        batchEnd = min(t+opts.batchSize-1, numel(subset));
        batch = subset(batchStart : opts.numSubBatches : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        [inputs,labels] = state.getBatch(imdb, batch) ;
        
        if numel(opts.gpus) >= 1
            inputs = gpuArray(inputs);
            labels = gpuArray(labels);
        end
        
        if strcmp(mode, 'train')
            dzdy     = single(1);
            evalMode = 'normal';% forward and backward
        else
            dzdy     = [] ;
            evalMode = 'test';  % forward only
        end
        
        net.layers{end}.class = labels ;
        res = vl_simplenn(net, inputs, dzdy, res, ...
            'accumulate', s ~= 1, ...
            'mode', evalMode, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'cudnn', opts.cudnn) ;
    end
    
    if strcmp(mode, 'train')
        [state, net] = params_updates(state, net, res, opts, opts.batchSize) ;
    end
    
    lossL2 = gather(res(end).x) ;
    
    %--------add your code here------------------------
    
    %--------------------------------------------------
    
    fprintf('%s: epoch %02d : %3d/%3d: loss: %4.4f \n', mode, state.epoch,  ...
        fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize),lossL2) ;
   % fprintf('loss: %4.4f \n', lossL2) ;
    
end


function [state, net] = params_updates(state, net, res, opts, batchSize)

% solver: Adam
for l=numel(net.layers):-1:1
    for j=1:numel(res(l).dzdw)
        
        if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
            
            % special case for learning bnorm moments
            thisLR = net.layers{l}.learningRate(j);
            net.layers{l}.weights{j} = vl_taccum(...
                1 - thisLR, ...
                net.layers{l}.weights{j}, ...
                thisLR / batchSize, ...
                res(l).dzdw{j}) ;
            
        else
            
            %  if   j == 1 && strcmp(net.layers{l}.type, 'bnorm')
            %         c = net.layers{l}.weights{j};
            %         net.layers{l}.weights{j} = clipping(c,mean(abs(c))/2);
            %  end
            
            thisLR = state.learningRate * net.layers{l}.learningRate(j);
            state.layers{l}.t{j} = state.layers{l}.t{j} + 1;
            t = state.layers{l}.t{j};
            alpha = thisLR;
            lr = alpha * sqrt(1 - opts.beta2^t) / (1 - opts.beta1^t);
            
            state.layers{l}.m{j} = state.layers{l}.m{j} + (1 - opts.beta1) .* (res(l).dzdw{j} - state.layers{l}.m{j});
            state.layers{l}.v{j} = state.layers{l}.v{j} + (1 - opts.beta2) .* (res(l).dzdw{j} .* res(l).dzdw{j} - state.layers{l}.v{j});
            
            % weight decay
            net.layers{l}.weights{j} = net.layers{l}.weights{j} -  thisLR * opts.weightDecay * net.layers{l}.weightDecay(j) * net.layers{l}.weights{j};
            
            % update weights
            net.layers{l}.weights{j} = net.layers{l}.weights{j} - lr * state.layers{l}.m{j} ./ (sqrt(state.layers{l}.v{j}) + opts.epsilon) ;
            
            
            %--------add your own code to update the parameters here-------
            
            if rand > 0.99
                A = net.layers{l}.weights{j};
                if numel(A)>=3*3*64
                    A = reshape(A,[size(A,1)*size(A,2)*size(A,3),size(A,4),1,1]);
                    if size(A,1)> size(A,2)
                        [U,S,V] = svd(A,0);
                    else
                        [U,S,V] = svd(A,'econ');
                    end
                    S1 =smallClipping2(diag(S),1.1,0.9);
                    A = U*diag(S1)*V';
                    A = reshape(A,size(net.layers{l}.weights{j}));
                    net.layers{l}.weights{j} = A;
                end
            end
            
            %--------------------------------------------------------------
            
        end
    end
end
%end


function epoch = findLastCheckpoint(modelDir,modelName)
list = dir(fullfile(modelDir, [modelName,'-epoch-*.mat'])) ;
tokens = regexp({list.name}, [modelName,'-epoch-([\d]+).mat'], 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;


function A = smallClipping(A, theta)
A(A>theta)  = A(A>theta) -0.000001;
A(A<-theta) = A(A<-theta)+0.000001;


function A = smallClipping2(A, theta1,theta2)
A(A>theta1)  = A(A>theta1)-0.00001;
A(A<theta2)  = A(A<theta2)+0.00001;


function fn = getBatch
fn = @(x,y) getSimpleNNBatch(x,y);



function [inputs,labels] = getSimpleNNBatch(imdb, batch)

global sigmas;
K = randi(8);
labels = imdb.HRlabels(:,:,:,batch);
labels = data_augmentation(labels,K);

sigma_max = 75;
sigmas = (rand(1,size(labels,4))*sigma_max)/255;
%sigmas   = round(7*chi2rnd(3,[1,size(labels,4)]))/255;
inputs = labels + bsxfun(@times,randn(size(labels)), reshape(sigmas,[1,1,1,size(labels,4)]));





