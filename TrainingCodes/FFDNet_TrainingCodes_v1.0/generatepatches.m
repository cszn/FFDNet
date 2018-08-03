function [imdb] = generatepatches

%% Note, set your training image set first, large dataset is prefered!
folders    = {'path_of_your_training_dataset'}; % set this first!

stride     = 60;  % control the number of image patches
patchsize  = 70;

batchSize  = 256; % important for BNorm
count      = 0;
nch        = 1;   % 1 for grayscale image, 3 for color image

step1      = 0;
step2      = 0;

ext               =  {'*.jpg','*.png','*.bmp'};
filepaths         =  [];

for j = 1:length(folders)
    for i = 1 : length(ext)
        filepaths = cat(1,filepaths, dir(fullfile(folders{j}, ext{i})));
    end
end

cshuffle = randperm(length(filepaths));
nimages  = round(length(filepaths)); % control the number of image patches

ns       = 1;
nscales  = min(1,0.45 + 0.05*randi(21,[ns,nimages]));
naugment = randi(8,[1,nimages]);

for i = 1 : nimages
    % HR = imread(fullfile(filepaths(cshuffle(i)).folder,filepaths(cshuffle(i)).name));
    HR = imread(fullfile(folders{1},filepaths(cshuffle(i)).name));
    HR = HR(:,:,1);
    
    HR = data_augmentation(HR, naugment(i));
    disp([i,nimages,round(count/batchSize)])
    
    for j = 1: size(nscales,1)
        
        HR_current  = imresize(HR,nscales(j,i),'bicubic');
        [hei,wid,~] = size(HR_current);
        for x = 1+step1 : stride : (hei-patchsize+1)
            for y = 1+step2 : stride : (wid-patchsize+1)
                count=count+1;
            end
        end
    end
end


numPatches  = ceil(count/batchSize)*batchSize;
diffPatches = numPatches - count;
disp([numPatches,numPatches/batchSize,diffPatches]);

disp('-----------------------------');

%------------------------------------------------------------------
%------------------------------------------------------------------

count = 0;
imdb.HRlabels  = zeros(patchsize, patchsize, nch, numPatches,'single');

for i = 1 : nimages
    % HR = imread(fullfile(filepaths(cshuffle(i)).folder,filepaths(cshuffle(i)).name));
    HR = imread(fullfile(folders{1},filepaths(cshuffle(i)).name));
    
    if nch ==1 && size(HR,3) == 3
        HR = rgb2gray(HR);
    end
    
    HR = data_augmentation(HR, naugment(i));
    disp([i,nimages,round(count/256)])
    
    for j = 1: size(nscales,1)
        
        HR_current  = imresize(HR,nscales(j,i),'bicubic');
        [hei,wid,~] = size(HR_current);
        HR_current  = im2single(HR_current);
        
        for x = 1+step1 : stride : (hei-patchsize+1)
            for y = 1+step2 : stride : (wid-patchsize+1)
                count = count + 1;
                subim_label  = HR_current(x : x+patchsize-1, y : y+patchsize-1,1:nch);
                imdb.HRlabels(:, :, :, count) = subim_label;
                if count<=diffPatches
                    imdb.HRlabels(:, :, :, end-count+1)   = HR_current(x : x+patchsize-1, y : y+patchsize-1,1:nch);
                end
            end
        end
    end
end

imdb.set    = uint8(ones(1,size(imdb.HRlabels,4)));

