function [mu,s,tau] = fitExGauss(rts)
% Finds the parameters of an ex-gaussian function given an rt distribution
% using maximum likelihood estimation.
% [mu,s,tau] = fitExGauss(rts)
%
% Input rts is an array of response times. For good results, this array
% should have at least 30 entries. An error is thrown if it has less than
% 2.
%
% Outputs:
%
% mu and s the mean and standard devaition for the gaussian part of the
% exgaussian.
% tau is the parameter of the exponential part of the exgaussian.
%
% Written by Benjamin Files.
%
% References
% Palmer et al., (2011) What are the Shapes of Response Time
% Distributions in Visual Search?,  Exp Psychol Hum Percept Perform.;
% 37(1): 58?71. doi:10.1037/a0020747
%
% Van Zandt, T. (2000). How to fit a response time distribution.
% Psychonomic Bulletin & Review, 7(3), 424-465.
%
% Inspired by DISTRIB toolbox of Yves Lacouture
% http://www.psy.ulaval.ca/?pid=1529
%

if numel(rts) < 3,
    error('Button:FitExGauss:NotEnoughSamples',...
        ['Cannot fit an ex-gaussian with less than 3 response times. ',...
        'only %d were provided.'],numel(rts));
end
if numel(rts) < 30,
    warning('Button:FitExGauss:FewSamples',...
        ['Fitting an ex-gaussian with %d samples. A fit will be ',...
        'provided, but the fit quality might be poor.'],numel(rts));
end

% These initial values are recommended in DISTRIB:
tauInit = std(rts)*.8;
muInit = mean(rts)-tauInit;
sigInit = sqrt(var(rts)-(tauInit.^2));

% Alternatively, use method of moments (e.g. Olivier, J., & Norberg, M. M.
% (2015). Positively Skewed Data: Revisiting the Box-Cox Power
% Transformation. International Journal of Psychological Research, 3(1),
% 68-77.
%
% This method is more accurate but also brittle (it has a lot of weird
% edges that result in nonsense estimates).  
% 
% g = skewness(rts);
% s = std(rts);
% m = mean(rts);
% 
% muInit = m-s*(g/2)^(1/3);
% sigInit = sqrt( s^2*(1-(g/2)^(2/3)) );
% tauInit = s*(g/2)^(1/3);


start = [muInit,sigInit,tauInit];

% Setting bounds appropriately is tricky. In particular, if we let tau get
% too small, we overflow, because tau appears in the denominator of the
% eventual expression of the exGaussian PDF. max([0.01, min(rts)]) is
% mostly from trial-and-error.
%
% Analternative approach (not implemented) might be to check if MLE is
% trying to use a very small tau and instead of erroring, instead default
% to a normal distribution.
lb = [min(rts) min(rts) max([0.01, min(rts)])]; 
ub = [max(rts) lb(2)+range(rts) lb(end)+range(rts)];

too_low = start < lb;
start(too_low) = lb(too_low)+eps;

too_high = start > ub;
start(too_high) = ub(too_high)-eps;

ss = statset(@mlecustom);
ss.MaxFunEvals = 200*numel(rts);
ss.MaxIter = 200*numel(rts);
ss.FunValCheck = 'on';
phat = mle(rts,'pdf',@rpe.exGaussPdf,'start',start,'lowerbound',...
    lb,'upperbound',ub,'options',ss);
mu = phat(1);
s = phat(2);
tau = phat(3);
end


% Copyright notice
%    Copyright 2016 Benjamin T. Files
% 
%    Licensed under the Apache License, Version 2.0 (the "License");
%    you may not use this file except in compliance with the License.
%    You may obtain a copy of the License at
% 
%        http://www.apache.org/licenses/LICENSE-2.0
% 
%    Unless required by applicable law or agreed to in writing, software
%    distributed under the License is distributed on an "AS IS" BASIS,
%    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
%    implied. See the License for the specific language governing
%    permissions and limitations under the License.