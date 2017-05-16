function attrib = att(t,si, pdf_fcn, stimulus_times, max_rt)
% Compute attribution onto a stimulus of interest given response(s) at
% time(s) t.
% attrib = att(t, si, pdf_fcn, stimulus_times, max_rt)
%
% Attribution is a normalized liklihood. Attribution of a
% response at time t(idx) is the likelihood that response was
% generated at time si divided by the sum of the liklihoods
% that response was generated at the time of all other possible
% evoking stimuli.
%
% t is a list of times at which a response could happen, si is
% the time at which a stimulus of interest occurs.
% compute attribution onto stimulus at time si for responses
% occurring at time t
% pdf_fcn is a function that takes one variable and returns a probability
% density for each value in that variable.
% stimulus_times is a vector of times at which stimuli were presented
% max_rt is the maximum response time supported

% compute the likelihood for the stimulus of interest
likelihood = zeros(size(t));
t_positive = t-si>=0;
likelihood(t_positive) = pdf_fcn(t(t_positive)-si);

% choose those stimuli that could get some attribution. These
% are needed for normalization
stim_subidx = stimulus_times >=(min(t)-max_rt) &...
    stimulus_times <= max(t);

other_stim = stimulus_times(stim_subidx);

% Compute attribution onto these other stimuli for
% normalization
attrib_other = zeros(numel(other_stim),numel(t));
for iOther = 1:numel(other_stim),
    t_positive_other = t-other_stim(iOther)>=0;
    attrib_other(iOther,t_positive_other) = pdf_fcn(...
        t(t_positive_other)-other_stim(iOther));
end
normalizer = sum(attrib_other,1);
attrib = likelihood./normalizer;
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