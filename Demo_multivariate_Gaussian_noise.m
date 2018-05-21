% This is the testing demo of FFDNet for denoising noisy color images corrupted by
% multivariate (3D) Gaussian noise model N([0,0,0]; Sigma) with zero mean and 
% covariance matrix Sigma in the RGB color space.
%
% To run the code, you should install Matconvnet first. Alternatively, you can use the
% function `vl_ffdnet_matlab` to perform denoising without Matconvnet.
%
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image
% Denoising" 2018/03/23
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

% clear; clc;

format compact;
global sigmas; % input noise level or input noise level map
addpath(fullfile('utilities'));

folderModel = 'models';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'BSD68','Set12','CBSD68','Kodak24','McMaster'}; % testing datasets
setTestCur  = imageSets{3};      % current testing dataset

showResult  = 1;
useGPU      = 1;
pauseTime   = 0;

%% Sigma
%% noise case 1 (general)
%   [(sigma_R)^2 &    sigma_RG     & sigma_RB 
%    sigma_RG   &   (sigma_G)^2   & sigma_GB
%    sigma_RB   &    sigma_GB     & (sigma_B)^2]

%% noise case 2 (special) channel-independent noise,  the widely-used noise model for color image
%   [(sigma)^2   &    0            &  0
%    0           &   (sigma)^2     &  0
%    0           &    0            & (sigma)^2]

%% noise case 3 (special) channel-independent noise
%   [(sigma_R)^2 &    0            &  0
%    0           &   (sigma_G)^2   &  0
%    0           &    0            & (sigma_B)^2]

%% noise case 4 (special) each channel has the same noise,  the widely-used noise model for grayscale image
%   [(sigma)^2   &   (sigma)^2     &  (sigma)^2
%    (sigma)^2   &   (sigma)^2     &  (sigma)^2
%    (sigma)^2   &   (sigma)^2     & (sigma)^2]


% image noise Sigma
noisecase = 2;

switch noisecase
    case 1
        L = 75/255;
        D = diag(rand(3,1));
        U = orth(rand(3,3));
        imageNoiseSigma = abs(L^2*(U' * D * U));
    case 2
        noiseLevelRGB = 50;
        imageNoiseSigma = (noiseLevelRGB/255)^2*eye(3);
    case 3
        noiseLevelR = 15;
        noiseLevelG = 30;
        noiseLevelB = 45;
        imageNoiseSigma = [(noiseLevelR/255)^2, 0, 0; 0, (noiseLevelG/255)^2, 0; 0, 0, (noiseLevelB/255)^2];
    case 4 
        noiseLevel = 50;
        imageNoiseSigma = (noiseLevel/255)^2*ones(3);
end

assert(sum(sum(imageNoiseSigma<0)) == 0,'We assume that all the elements are non-negative.');

% input noise Sigma
inputNoiseSigma =  imageNoiseSigma;
sigma           =  sqrt(inputNoiseSigma);
sigma           =  sigma(:);
sigma6          =  sigma([1,4,5,7,8,9]); % extract 6 parameters


folderResultCur       =  fullfile(folderResult, [setTestCur,'_',num2str(imageNoiseSigma(1,1)),'_',num2str(inputNoiseSigma(1,1))]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

load(fullfile('models','FFDNet_3D_RGB.mat'));
net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

% read images
ext         =  {'*.jpg','*.png','*.bmp','*.tif'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end

% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    
    % read images
    label   = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,c] = size(label);
    label   = im2double(label);
    if c == 1
        label = cat(3,label,label,label);
    end
    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    
    % add noise
    randn('seed',0);
    % noise = bsxfun(@times,randn(size(label)),permute(imageNoiseSigma/255,[3 4 1 2]));
    
    noise = vl_randnm(w, h, imageNoiseSigma);
    
    input = single(label + noise);
    
    if mod(w,2)==1
        input = cat(1,input, input(end,:,:)) ;
    end
    if mod(h,2)==1
        input = cat(2,input, input(:,end,:)) ;
    end
    
    % tic;
    if useGPU
        input = gpuArray(input);
    end
    
    % set noise level map
    sigmas = sigma6; % see "vl_simplenn.m".
    
    % perform denoising
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
    % res    = vl_ffdnet_concise(net, input);    % concise version of vl_simplenn for testing FFDNet
    %res    = vl_ffdnet_matlab(net, input); % use this if you did  not install matconvnet; very slow
    
    % output = input -res(end).x; % for 'model_color.mat'
    output = res(end).x;
    
    
    if mod(w,2)==1
        output = output(1:end-1,:,:);
        input  = input(1:end-1,:,:);
    end
    if mod(h,2)==1
        output = output(:,1:end-1,:);
        input  = input(:,1:end-1,:);
    end
    
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    %toc;
    
    if c == 1
        output= mean(output,3);
        input = mean(input,3);
        label = mean(label,3);
    end
    
    % calculate PSNR, SSIM and save results
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showResult
        imshow(cat(2,im2uint8(input),im2uint8(label),im2uint8(output)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        %imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' num2str(imageNoiseSigma(1,1),'%02d'),'_' num2str(inputNoiseSigma(1,1),'%02d'),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), extCur] ));
        drawnow;
        pause(pauseTime)
    end
    disp([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
    
end

disp([mean(PSNRs),mean(SSIMs)]);




