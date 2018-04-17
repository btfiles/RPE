% Estimates performance on RSVP target detection tasks
%
% This package contains code implementing the regression method of Files &
% Marathe (2016) for estimating performance on rapid serial visual
% presentation target detection experiments.  Performance is summarized as
% hit rate (HR) and false alarm rate (FAR).
%
% There is also an implementation of the maximum likelihood estimator
% described in Canady, Marathe, Herman & Files (2018). The MLE method
% is somewhat more accurate than the regression method, but takes longer to
% run.
%
% The package also includes code implementing both a maximum a-posteriori
% method and a fully Bayesian method that returns distribution estimates of
% all performance parameters. The MAP method works about as well as the MLE
% method as long as the priors are reasonable. The fully Bayesian method
% uses adaptive Metropolis sampling to approximate the posterior
% distribution. It is incredibly slow and is not well-tested, so its use is
% not encouraged.
%
% This package should be distributed with examples (example_script.m,
% example_script_mle.m, example_script_bayes.m) that illustrate its use.
%
% Example
%   e = rpe.RSVPPerformanceEstimator(stim_time, stim_lbl, button_time);
%   [hr, far] = e.runEstimates; % may take a few seconds to several minutes
%
% Files
%   exGaussPdf               - a probability density function for the exgaussian distribution.
%   fitExGauss               - Finds the parameters of an ex-gaussian function given an rt distribution
%   RSVPPerformanceEstimator - An implementation of a regression-based method for estimating hit
%   RSVPPerformanceBayes     - A very slow Bayesian estimator of RSVP performance
%   RSVPPerformanceMAP       - Implements maximum likelihood and maximum a-posteriori estimates of RSVP performance
%
% Reference
% Files, B. T., & Marathe, A. R. (2016). A regression method for estimating
% performance in a rapid serial visual presentation target detection task.
% Journal of Neuroscience Methods, 258, 114?123.
% http://doi.org/10.1016/j.jneumeth.2015.11.003

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