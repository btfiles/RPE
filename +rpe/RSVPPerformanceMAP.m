classdef RSVPPerformanceMAP < rpe.RSVPPerformanceBayes
    % Maximum a-posteriori estimation
    
    properties
        use_prior = true
    end
    methods
        function obj = RSVPPerformanceMAP(varargin)
            obj@rpe.RSVPPerformanceBayes(varargin{:});
        end

        
        function [HR, FAR, ef] = estimatePerformance(obj)
            % Estimates performance on a RSVP target detection experiment.
            % [HR, FAR] = estimatePerformance(obj)
            %
            % Uses maximum a-posteriori estimation to select the HR, FAR,
            % and ex-gaussian parameters that result in the highest
            % probability of obtained results.
            %
            % Here, results means the list of times at which a button was
            % pressed. The generative model treats times of stimulus onset
            % and their labels as fixed.
            %
            % [HR, FAR, EF] = estimatePerformance(...) also returns an exit
            % flag from fminsearch.
            %
            % see also fminsearch
            
            % Work out time bins of width time_resolution (a property of
            % this class) and place responses into their bins
            t_min = min([min(obj.stimulus_times), min(obj.buttonpress_times)]);
            t_max = max([max(obj.stimulus_times), max(obj.buttonpress_times)]) + obj.pdf_support;
            
            obj.t = t_min:obj.time_resolution:t_max;
            
            bpidx = floor((obj.buttonpress_times-t_min)/obj.time_resolution) + 1;
            bp = false(size(obj.t));
            bp(bpidx) = true;
            
            %N = far*NT + hr*T
            if isempty(obj.hrfun)
                N = numel(obj.buttonpress_times);
                T = sum(obj.stimulus_labels==1);
                NT = sum(obj.stimulus_labels==0);
                obj.hrfun = @(hr) (N-hr*T)/NT;
            end
            
            if obj.use_prior
                objective = @(theta) -1*(obj.xloglikelihood(theta(1), theta(2:end), bp) + log(obj.prior(theta)));
            else
                objective = @(theta) -1*obj.xloglikelihood(theta(1), theta(2:end), bp);
                warning('Not using priors. This is MLE, not MAP.');
            end
            
            % Sample several initial values.
            ninit = 10;
            nretry = 3;
            tic
            retry_count = 0;
            
            % Try to find some initial values that do not have 0
            % likelihood.
            
            while retry_count < nretry
                inits = zeros(ninit, 4);
                res = zeros(ninit, 1);
                for i = 1:ninit
                    inits(i,:) = [obj.prs_h, obj.prs_mu, obj.prs_sig, obj.prs_tau];
                    res(i) = objective(inits(i, :));
                end
                if ~all(isinf(res))
                    break
                else
                    retry_count = retry_count+1;
                end
            end
            if all(isinf(res))
                error('could not find a starting spot with lik~=0.');
            end
            [~, idx] = min(res);
            thminit = inits(idx, :);
            
            fprintf(1, 'Initial values: HR=%0.3f, mu=%0.3f, sig=%0.3f, tau=%0.3f\n', ...
                obj.logistic(thminit(1)), exp(thminit(2:end)));
            fprintf(1, 'Initial objective: %f\n', objective(thminit));
            
            fprintf(1, 'log likelihood: %f\n', obj.xloglikelihood(thminit(1), thminit(2:end), bp));
            fprintf(1, 'log prior: %f\n', log(obj.prior(thminit)));
            
            % now do minimization on the -log(posterior)
            opt = optimset('fminsearch');
            opt.UseParallel = false; % internals are parallel
            opt.Display = 'notify';
            t0 = tic;
            [X, fval, ef] = fminsearch(objective, thminit, opt);
            fprintf(1, 'Optimization took %f s.\n', toc(t0));
            fprintf(1, 'Final value was %f.\n', fval);
            if ef < 1
                warning('Exit flag of fminsearch was %d. Results may not be optimal.\n', ef);
            end
            
            if obj.diag
                obj.plotPriors()
            end
            
            HR = obj.logistic(X(1));
            FAR = obj.hrfun(HR);
            obj.setPdf(exp(X(2:end)));
        end
    end
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