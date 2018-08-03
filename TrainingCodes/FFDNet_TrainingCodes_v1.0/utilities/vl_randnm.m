function [noise,sigma6] = vl_randnm(w, h, Sigma)

% generate noise
mu     =  [0,0,0];
noise  =  mvnrnd(mu,Sigma,w*h);
noise  =  reshape(noise,[w,h,3]);

% extract 6 parameters
sigma  =  sqrt(Sigma);
sigma  =  sigma(:);
sigma6 =  sigma([1,4,5,7,8,9]);

end

