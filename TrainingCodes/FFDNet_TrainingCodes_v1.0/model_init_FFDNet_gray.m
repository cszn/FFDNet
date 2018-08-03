
function net = model_init_FFDNet_gray


lr11  = [1 1];
lr10  = [1 0];
weightDecay = [1 1];
nCh = 64; % number of channels
fSz = 3;  % fize size
nNm = 1;  % number of noise level map

useBnorm  = 0; % if useBnorm  = 0, you should also use adam.
nsubimage = 4; % 4 for grayscale image, 12 for color image

% Define network

net.layers = {} ;

net.layers{end+1} = struct('type', 'SubP','scale',1/2) ;

net.layers{end+1} = struct('type', 'concat') ;

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{orthrize2(sqrt(2/(9*nCh))*randn(fSz,fSz,nsubimage+nNm,nCh,'single')),  zeros(nCh,1,'single')}}, ...
    'stride', 1, ...
    'pad', 1, ...
    'dilate',1, ...
    'learningRate',lr11, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;

net.layers{end+1} = struct('type', 'relu','leak',0) ;

for i = 1:1:13
    
    if useBnorm == 0
        
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{orthrize2(sqrt(2/(9*nCh))*randn(fSz,fSz,nCh,nCh,'single')), zeros(nCh,1,'single')}}, ...
            'stride', 1, ...
            'learningRate',lr11, ...
            'dilate',1, ...
            'weightDecay',weightDecay, ...
            'pad', 1, 'opts', {{}}) ;
        
    else
        
        net.layers{end+1} = struct('type', 'conv', ...
            'weights', {{orthrize2(sqrt(2/(9*nCh))*randn(fSz,fSz,nCh,nCh,'single')), zeros(nCh,1,'single')}}, ...
            'stride', 1, ...
            'learningRate',lr10, ...
            'dilate',1, ...
            'weightDecay',weightDecay, ...
            'pad', 1, 'opts', {{}}) ;
        
        net.layers{end+1} = struct('type', 'bnorm', ...
            'weights', {{ones(nCh,1,'single'), zeros(nCh,1,'single'),[zeros(nCh,1,'single'), ones(nCh,1,'single')]}}, ...
            'learningRate', [1 1 1], ...
            'weightDecay', [0 0]) ;
        
    end
    
    net.layers{end+1} = struct('type', 'relu','leak',0) ;
    
end


net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{orthrize2(sqrt(2/(9*nCh))*randn(fSz,fSz,nCh,nsubimage,'single')), zeros(nsubimage,1,'single')}}, ...
    'stride', 1, ...
    'learningRate',lr10, ...
    'dilate',1, ...
    'weightDecay',weightDecay, ...
    'pad', 1, 'opts', {{}}) ;

net.layers{end+1} = struct('type', 'SubP','scale',2) ;


net.layers{end+1} = struct('type', 'loss') ; % make sure the new 'vl_nnloss.m' is in the same folder.

% Fill in default values
net = vl_simplenn_tidy(net);

end


function W = orthrize2(a)

s_ = size(a);
a = reshape(a,[size(a,1)*size(a,2)*size(a,3),size(a,4),1,1]);
[u,d,v] = svd(a, 'econ');
if(size(a,1) < size(a, 2))
    u = v';
end
%W = sqrt(2).*reshape(u, s_);
W = reshape(u, s_);

end


function A = clipping2(A,b)

A(A<b(1)) = b(1);
A(A>b(2)) = b(2);

end



function A = clipping(A,b)

A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;

end


