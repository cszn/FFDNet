function I = data_augmentation(I, K)


if K == 1
    return;
elseif K == 2 % flipped
    I = flipud(I);
    return;
elseif K == 3 % rotation 90
    I = rot90(I,1);
    return;
elseif K == 4 % rotation 90 & flipped
    I = rot90(I,1);
    I = flipud(I);
    return;
elseif K == 5 % rotation 180
    I = rot90(I,2);
    return;
elseif K == 6 % rotation 180 & flipped
    I = rot90(I,2);
    I = flipud(I);
    return;
elseif K == 7 % rotation 270
    I = rot90(I,3);
    return;
elseif K == 8 % rotation 270 & flipped
    I = rot90(I,3);
    I = flipud(I);
    return;
end






