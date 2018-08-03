function Y = vl_nnloss(X,c,dzdy,varargin)

% --------------------------------------------------------------------
% pixel-level L2 loss
% --------------------------------------------------------------------

d = X-c;
if nargin <= 2 || isempty(dzdy)
    t = (d.^2)/2;
    Y = sum(t(:))/size(X,4); 
else
    Y = d;
end

% --------------------------------------------------------------------

% if nargin <= 2 || isempty(dzdy)
%     t = ((X-c).^2)/2;
%     Y = sum(t(:))/size(X,4); % reconstruction error per sample;
% else
%     Y = sign(X - c);
% end

% --------------fine-tune---------------------------------------------

% global sigmas;
% d = X-c;
% w = min(200,1./((sigmas/75*255).^1.2));
% if nargin <= 2 || isempty(dzdy)
%     t = (d.^2)/2;
%     Y = sum(t(:))/size(X,4);
% else
%     Y = bsxfun(@times,d,reshape(w,[1,1,1,size(X,4)]));
% end

% --------------------------------------------------------------------

% eps = 1e-5;
% d = X - c;
% e = sqrt(d.^2 + eps);
% 
% if nargin <= 2 || isempty(dzdy)
%     t = (d.^2)/2;
%     Y = sum(t(:))/size(X,4); 
% else
%     Y = d ./ e;
% end
