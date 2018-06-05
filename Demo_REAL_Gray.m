% This is the testing demo of FFDNet for denoising real noisy grayscale images.
%
% To run the code, you should install Matconvnet first. Alternatively, you can use the
% function `vl_ffdnet_matlab` to perform denoising without Matconvnet.
%
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"
%  2018/03/23
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

% clear; clc;
format compact;
global sigmas; % input noise level or input noise level map
addpath(fullfile('utilities'));

folderModel = 'models';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'RNI6'};         % testing datasets
setTestCur  = imageSets{1};      % current testing dataset

showResult  = 1;
useGPU      = 1;
pauseTime   = 0;


inputNoiseSigma = 15;  % input noise level
% -****************************************************-
% Building.png        (inputNoiseSigma = 20); i = 1
% Chupa_Chups.png     (inputNoiseSigma = 10); i = 2
% David_Hilbert.png   (inputNoiseSigma = 15); i = 3
% Marilyn.png         (inputNoiseSigma = 7);  i = 4
% Old_Tom_Morris.png  (inputNoiseSigma = 15); i = 5
% Vinegar.png         (inputNoiseSigma = 20); i = 6
% -****************************************************-

folderResultCur       =  fullfile(folderResult, [setTestCur,'_',num2str(inputNoiseSigma)]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

load(fullfile('models','FFDNet_gray.mat'));
net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end


for i = 1
    
    % read images
    disp([filePaths(i).name])
    label = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,~]=size(label);
    if size(label,3)==3
        label = rgb2gray(label);
    end
    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    input = im2single(label);
    
    if mod(w,2)==1
        input = cat(1,input, input(end,:)) ;
    end
    if mod(h,2)==1
        input = cat(2,input, input(:,end)) ;
    end
    
    % tic;
    if useGPU
        input = gpuArray(input);
    end
    
    % set noise level map
    sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
    
    % perform denoising
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
    % res    = vl_ffdnet_concise(net, input);    % concise version of vl_simplenn for testing FFDNet
    % res    = vl_ffdnet_matlab(net, input); % use this if you did  not install matconvnet; very slow
    
    % output = input - res(end).x; % for 'model_gray.mat'
    output = res(end).x;
    
    if mod(w,2)==1
        output = output(1:end-1,:);
        input  = input(1:end-1,:);
    end
    if mod(h,2)==1
        output = output(:,1:end-1);
        input  = input(:,1:end-1);
    end
    
    % convert to CPU
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    % toc;
    if showResult
        imshow(cat(2,im2uint8(input),im2uint8(output)));
        title([filePaths(i).name])
        imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' num2str(inputNoiseSigma,'%02d'), '.png'] ));
        drawnow;
        pause(pauseTime)
    end
    
    
end




