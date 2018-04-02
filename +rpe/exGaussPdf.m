function p =exGaussPdf(x,mu,s,tau)
%EXGAUSSPDF a probability density function for the exgaussian distribution.
%p = exGaussPdf(x,mu,s,tau) 
%
%Mu, s and tau should be real numbers not less than zero.  Throws an error
%if not. 
%Note, this blows up if tau is too small. 
%x is the time(s) for which the probability density is requested. 
%mu and s are the mean and standard devaition for the gaussian part of the
%exgaussian.
%tau is the parameter of the exponential part of the exgaussian. 
%x, mu, s and tau are all assumed to have the same units.
%
% Written by Benjamin Files

% validate input.
ip = inputParser;
ip.addRequired('x',@isnumeric);
ip.addRequired('mu',@checkInput);
ip.addRequired('s',@checkInput);
ip.addRequired('tau',@checkInput);
ip.parse(x,mu,s,tau);
x = ip.Results.x;
mu = ip.Results.mu;
s = ip.Results.s;
tau = ip.Results.tau;

% check for overflow FIXME -- maybe switch to normal if tau is small?
% tmp = mu/tau + s^2/(2*tau.^2) - x/tau;
% maxtmp = log(realmax);
% if any(tmp>=maxtmp),
%     warning('RPE:ExGaussPdf:BigExpPart',...
%         'A value exceeded max allowed.  Results will be approximate.');
%     tmp(tmp>=maxtmp) = maxtmp/100;
% end

% compute the pdf
% pE = exp(tmp);
% pG = normcdf( (x - mu - (s.^2/tau))/abs(s));
% p = (1/tau).*pE.*pG;

% Do computations in log units, convert to prob at the end
lpE = mu/tau + s^2/(2*tau.^2) - x/tau;
lpG = log(normcdf( (x - mu - (s.^2/tau))/abs(s)));
logp = lpE+lpG-log(tau);
    
if any(isinf(logp))
    warning('RPE:ExGaussPdf:Underflow', 'The value of tau is too small relative to s^2. Switching to normal approximation.');
    p = normpdf(x, mu, s);
else
    p = exp(logp);
end

if any (p<=eps)
    p(p<=eps) = eps;
end
end

function ok = checkInput(v)
checks = [isnumeric(v),~isnan(v),~isinf(v),v>0];
if any(~checks)
    warning('Failed check %d!', find(~checks, 1));
end
ok = all(checks);
end


% Copyright notice
%    Copyright 2018 Benjamin T. Files
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