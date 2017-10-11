%% This is the testing demo of FFDNet for denoising real noisy (color) images.
%% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"
%% If you have any question, please feel free to contact with me.
%% Kai Zhang (e-mail: cskaizhang@gmail.com)

% clear; clc;
format compact;
global sigmas;
addpath(fullfile('utilities'));

folderModel = 'models';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'RNI15'}; % testing datasets
setTestCur  = imageSets{1}; % current testing dataset

showResult  = 1;
useGPU      = 0;
pauseTime   = 0;

inputNoiseSigma = [15;15;15];  % input noise level (Sigma_R;Sigma_G;Sigma_B)
%% -****************************************************-
%% Audrey_Hepburn.jpg (inputNoiseSigma = 10); i = 1
%% Bears.png          (inputNoiseSigma = 15); i = 2
%% Boy.png            (inputNoiseSigma = 45); i = 3
%% Dog.png            (inputNoiseSigma = 28); i = 4
%% Flowers.png        (inputNoiseSigma = 70); i = 5
%% Frog.png           (inputNoiseSigma = 15); i = 6
%% Movie.png          (inputNoiseSigma = 12); i = 8
%% Pattern1.png       (inputNoiseSigma = 12); i = 9
%% Pattern2.png       (inputNoiseSigma = 45); i = 10
%% Pattern3.png       (inputNoiseSigma = 25); i = 11
%% Postcards.png      (inputNoiseSigma = 15); i = 12
%% Singer.png         (inputNoiseSigma = 30); i = 13
%% Stars.png          (inputNoiseSigma = 18); i = 14
%% Window.png         (inputNoiseSigma = 15); i = 15
%% -****************************************************-

folderResultCur       =  fullfile(folderResult, [setTestCur,'_',num2str(inputNoiseSigma(1))]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

load(fullfile('models','model_color.mat'));
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


for i = 6
    
    %% read images
    disp([filePaths(i).name])
    label = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,c]=size(label);
    
    if c == 3
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        
        input = im2single(label);
        
        if mod(w,2)==1
            input = cat(1,input, input(end,:,:)) ;
        end
        if mod(h,2)==1
            input = cat(2,input, input(:,end,:)) ;
        end
        
        if useGPU
            input = gpuArray(input);
        end
        
        %% set model noise level map
        sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
        
        %% do the denoising
        res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        output = input -res(end).x;
        
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
        
        if showResult
            imshow(cat(2,im2uint8(input),im2uint8(output)));
            title([filePaths(i).name])
            %imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur,'_' num2str(inputNoiseSigma(1),'%02d'), extCur] ));
            drawnow;
            pause(pauseTime)
        end
        
        
    end
end






