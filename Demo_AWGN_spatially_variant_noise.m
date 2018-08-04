% This is the testing demo of FFDNet for denoising noisy grayscale images corrupted by
% spatially variant AWGN.
%
% To run the code, you should install Matconvnet first. Alternatively, you can use the
% function `vl_ffdnet_matlab` to perform denoising without Matconvnet.
%
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"
%  2018/08/04
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

%clear; clc;
format compact;
global sigmas; % input noise level or input noise level map
addpath(fullfile('utilities'));

folderModel = 'models';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'BSD68','Set12'}; % testing datasets
setTestCur  = imageSets{2};      % current testing dataset

showResult  = 1;
useGPU      = 1; % CPU or GPU. For single-threaded (ST) CPU computation, use "matlab -singleCompThread" to start matlab.
pauseTime   = 0;

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

% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));

for i = 5
    
    % read images
    label = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,~]=size(label);
    if size(label,3)==3
        label = rgb2gray(label);
    end
    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    label = im2double(label);
    
    % noise level map
    [~,~,noiseSigma] = peaks(size(label,1));
    noiseSigma = 0  + (50 - 0).*(noiseSigma - min(noiseSigma(:)))./(max(noiseSigma(:)) - min(noiseSigma(:)));
    noiseSigma = data_augmentation(noiseSigma, 2);
    sigmas = imresize(noiseSigma,1/2,'bicubic')/255;
    
    % add noise
    randn('seed',1);
    noise = noiseSigma/255.*randn(size(label));
    input = single(label + noise);
    
    if mod(w,2)==1
        input = cat(1,input, input(end,:)) ;
    end
    if mod(h,2)==1
        input = cat(2,input, input(:,end)) ;
    end
    
    if useGPU
        input = gpuArray(input);
    end
    
    % perform denoising
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
    % res    = vl_ffdnet_concise(net, input);    % concise version of vl_simplenn for testing FFDNet
    % res    = vl_ffdnet_matlab(net, input); % use this if you did  not install matconvnet; very slow
    output = res(end).x;
    
    
    if mod(w,2)==1
        output = output(1:end-1,:);
        input  = input(1:end-1,:);
    end
    if mod(h,2)==1
        output = output(:,1:end-1);
        input  = input(:,1:end-1);
    end
    
    if useGPU
        output = gather(output);
        input  = gather(input);
    end

    % calculate PSNR, SSIM and save results
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showResult
        imshow(cat(2,im2uint8(input),im2uint8(label),im2uint8(output)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        %imwrite(im2uint8(output), '05_sv.png');
        drawnow;
        pause(pauseTime)
    end
    disp([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
end

disp([mean(PSNRs(i)),mean(SSIMs(i))]);

