function y = vl_nnSubP(x, dzdy, varargin)


opts.scale = 2;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;
scale = opts.scale;

if scale > 1 % many (small)---> one (big) e.g., 100X100X256X128  ----> 200X200X64*128
    scale2 = scale^2;
    if nargin <= 1 || isempty(dzdy)
        [hei,wid,channelI,bsize] = size(x);
        channelO = channelI/scale2;
        channelI_start = 0;
        idx1 = scale * (1:hei);
        idx2 = scale * (1:wid);
        y = zeros([scale*hei, scale*wid, channelO, bsize],'like',x);
        for nchannelO = 1:channelO
            for i = 1:scale2
                a = mod(i-1, scale) + 1;
                b = floor((i-1)/scale) + 1;
                y(idx1+a-scale, idx2+b-scale, nchannelO, :) = x(:,:,channelI_start+i,:);
            end
            channelI_start = scale2 + channelI_start;
        end
    else
        [hei,wid,channelO,bsize] = size(dzdy);
        idx1 = 1:scale:hei;
        idx2 = 1:scale:wid;
        channelI = channelO*scale2;
        channelI_start = 0;
        y = zeros([hei/scale, wid/scale, channelI, bsize],'like',x);
        for  nchannelO = 1:channelO
            for i = 1:scale^2
                a = mod(i-1, scale) + 1;
                b = floor((i-1)/scale) + 1;
                y(:, :, channelI_start+i, :) = dzdy(idx1+a-1, idx2+b-1,nchannelO,:);
            end
            channelI_start = scale2 + channelI_start;
        end
    end
    
else %  one (big) ---> many (small)  e.g., 200X200X64*128--->100X100X256X128
    
    scale = round(1/scale);
    scale2 = scale^2;
    if nargin <= 1 || isempty(dzdy)
        [hei,wid,channelO,bsize] = size(x);
        idx1 = 1:scale:hei;
        idx2 = 1:scale:wid;
        channelI = channelO*scale2;
        channelI_start = 0;
        y = zeros([hei/scale, wid/scale, channelI, bsize],'like',x);
        for  nchannelO = 1:channelO
            for i = 1:scale^2
                a = mod(i-1, scale) + 1;
                b = floor((i-1)/scale) + 1;
                y(:, :, channelI_start+i, :) = x(idx1+a-1, idx2+b-1,nchannelO,:);
            end
            channelI_start = scale2 + channelI_start;
        end
    else
        [hei,wid,channelI,bsize] = size(dzdy);
        channelO = channelI/scale2;
        channelI_start = 0;
        idx1 = scale * (1:hei);
        idx2 = scale * (1:wid);
        y = zeros([scale*hei, scale*wid, channelO, bsize],'like',x);
        for nchannelO = 1:channelO
            for i = 1:scale2
                a = mod(i-1, scale) + 1;
                b = floor((i-1)/scale) + 1;
                y(idx1+a-scale, idx2+b-scale, nchannelO, :) = dzdy(:,:,channelI_start+i,:);
            end
            channelI_start = scale2 + channelI_start;
        end
    end
end

