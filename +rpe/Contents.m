% Estimates performance on RSVP target detection tasks
%
% This package contains code implementing the regression method of Files &
% Marathe (2016) for estimating performance on rapid serial visual
% presentation target detection experiments.  Performance is summarized as
% hit rate (HR) and false alarm rate (FAR).
%
% This package should be distributed with an example (example_script.m)
% that illustrates its use.
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