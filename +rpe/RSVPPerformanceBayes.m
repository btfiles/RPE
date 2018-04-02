classdef RSVPPerformanceBayes < rpe.RSVPPerformanceEstimator
    % Implements a fully-Bayesian solution based on MCMC sampling from the
    % posterior. In particular, it uses adaptive Metropolis sampling to
    % generate draws from the posteriors. This is extremely computationally
    % intensive relative to MLE and MAP solutions, and it has not been
    % extensively tested. Unless some properties of the posterior
    % distribution are of special interest, it is recommended to stick with
    % the MLE and MAP solutions in rpe.RSVPPerformanceMAP
    %
    
    properties
        %Hyper-parameters of exgaussian (lognormal)
        hp_mu = [-1.5, 0.4];
        hp_mu_delt = 0.00005;
        hp_sig = [-1.5, 0.3];
        hp_sig_delt = 0.00005;
        hp_tau = [-2.2, 1];
        hp_tau_delt = 0.00005;
        
        
        % Hyper-parameters of hit rate (beta)
        hp_h = [2 .65];
        hp_h_delt = 0.0001;
        
        diag = false; % set to true for diagnostic plots
        
        %Settings for MWG sampling
        samp_per_chain = 10000;
        warmup_per_chain = 10000;
        batch_update = 100;
        
    end
    
    properties
        hrfun = [];
        trace = [];
        trace_lik = [];
        trace_prior = [];
    end
    
    %% Priors
    methods
        function p = pr_mu(obj, mu)
            p = lognpdf(exp(mu), obj.hp_mu(1), obj.hp_mu(2)).*obj.hp_mu_delt;
        end
        function s = prs_mu(obj)
            s = normrnd(obj.hp_mu(1), obj.hp_mu(2));
        end
        function p = pr_sig(obj, sig)
            p = lognpdf(exp(sig), obj.hp_sig(1), obj.hp_sig(2)).*obj.hp_sig_delt;
        end
        function s = prs_sig(obj)
            s = normrnd(obj.hp_sig(1), obj.hp_sig(2));
        end
        function p = pr_tau(obj, tau)
            p = lognpdf(exp(tau), obj.hp_tau(1), obj.hp_tau(2)).*obj.hp_tau_delt;
        end
        function s = prs_tau(obj)
            s = normrnd(obj.hp_tau(1), obj.hp_tau(2));
        end
        function p = pr_h(obj, h)
            h = obj.logistic(h);
            alpha = obj.hp_h(1)*obj.hp_h(2);
            beta = obj.hp_h(1)*(1-obj.hp_h(2));
            p = betapdf(h, alpha, beta).*obj.hp_h_delt;
        end
        function s = prs_h(obj)
            alpha = obj.hp_h(1)*obj.hp_h(2);
            beta = obj.hp_h(1)*(1-obj.hp_h(2));
            s = betarnd(alpha, beta);
            s = obj.logit(s);
        end
        function lp = lprior(obj, theta)
            lp = sum(log([...
                obj.pr_h(theta(1)), ...
                obj.pr_mu(theta(2)), ...
                obj.pr_sig(theta(3)), ...
                obj.pr_tau(theta(4))]));
        end
        function p = prior(obj, theta)
            p = exp(lprior(obj, theta));
        end
        function l = likelihood(obj, hr, exg, bp)
            hr = obj.logistic(hr);
            far = obj.hrfun(hr);
            exg = exp(exg);
            
            if hr>1 || hr<0 || far>1 || far<0 || any(exg<0)
                l = 0;
            else
                l = exp(obj.logLikelihood(hr, far, exg, bp));
            end
        end
    end
    
    methods
        function obj = RSVPPerformanceBayes(varargin)
            obj@rpe.RSVPPerformanceEstimator(varargin{:});
        end
        function [HR, FAR] = estimatePerformance(obj)
            % Estimates performance on a RSVP target detection experiment.
            % [HR, FAR] = estimatePerformance(obj)
            % Here, results means the list of times at which a button was
            % pressed. The generative model treats times of stimulus onset
            % and their labels as fixed.
            
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
            
            llik = @(theta) obj.xloglikelihood(theta(1), theta(2:end), bp);
            lpr = @(theta) obj.lprior(theta);
            for i = 1:100
                thminit = [obj.prs_h, obj.prs_mu, obj.prs_sig, obj.prs_tau];
                if llik(thminit)~=0
                    break
                end
            end
            if i==100
                error('could not find a starting spot with lik~=0.');
            end
            mwg = rpe.MWG(ones(1,4).*.2, llik, lpr, thminit);
            
            mwg.nmh = obj.samp_per_chain;
            mwg.nbi = obj.warmup_per_chain;
            mwg.nbatch = obj.batch_update;
            
            t0 = tic;
            
            g = gcp();
            n_chain = g.NumWorkers;
            
            parfor i = 1:n_chain
                [thetac{i}, thlikc{i}, thpriorc{i}] = mwg.sample();
            end
            theta = cat(1, thetac{:});
            obj.trace = theta;
            obj.trace_lik = cat(1, thlikc{:});
            obj.trace_prior = cat(1, thpriorc{:});
            fprintf(1, 'Sampling took %f s.\n', toc(t0));
            
            if obj.diag
                obj.plotPriors()
                obj.plotPosteriors(theta)
            end
            
            mtheta = mean(theta);
            HR = obj.logistic(mtheta(1));
            FAR = obj.hrfun(HR);
            obj.setPdf(exp(mtheta(2:end)));
        end
        function [bp, t] = buildTimeIdx(obj)
            % Computes sample times and converts button press times to
            % sample idx.
            
            t_min = min([min(obj.stimulus_times), min(obj.buttonpress_times)]);
            t_max = max([max(obj.stimulus_times), max(obj.buttonpress_times)]) + obj.pdf_support;
            
            t = t_min:obj.time_resolution:t_max;
            
            bpidx = floor((obj.buttonpress_times-t_min)/obj.time_resolution) + 1;
            bp = false(size(t));
            bp(bpidx) = true;
        end
        function llik = logLikelihood(obj, hr, far, exg, bp)
            % bp is a boolean array that is true for time bins with a
            % button press.
            
            % need time setup, if it isn't
            if isempty(obj.t)
                [~, obj.t] = obj.buildTimeIdx();
            end
            
            p_resp = obj.pResp(hr, far, exg);
            
            % eliminate zeros and ones
            p_resp(p_resp==0) = eps;
            p_resp(p_resp==1) = 1-eps;
            
            % doing this without the log results in numerical underflow.
            llik = sum(log(p_resp(bp))) + sum(log(1-p_resp(~bp)));
        end
        function l = xloglikelihood(obj, hr, exg, bp)
            % Transforms hr and exg back into native format then computes
            % log likelihood. See also RSVPPerformanceBayes.logLikelihood
            
            hr = obj.logistic(hr);
            far = obj.hrfun(hr);
            exg = exp(exg);
            
            if hr>1 || hr<0 || far>1 || far<0 || any(exg<=0)
                l = -inf;
            else
                l = obj.logLikelihood(hr, far, exg, bp);
            end
        end
    end
    
    %% Probability model
    methods
        function p = pResp(obj, hr, far, exg)
            % Computes the probability of responding in each time bin given
            % HR and FAR.
            % p_all = pResp(obj, hr, far, exg)
            % Note: I've tried to parallelize this or run it on the GPU,
            % but that always seems to slow it down instead of speeding it
            % up.

            t_init = obj.t(1);
            num_t = numel(obj.t);
            
            pdf = rpe.exGaussPdf(0:obj.time_resolution:obj.pdf_support, ...
                exg(1), exg(2), exg(3)).*obj.time_resolution;
            
            % response rate array: HR for targets, FAR for non-targets
            rra = far*ones(size(obj.stimulus_labels));
            rra(obj.stimulus_labels==true) = hr;
            
            %% put stimuli in the same time bins as the responses.
            stim_idx = round((obj.stimulus_times-t_init)/obj.time_resolution) + 1;
            
            %% Ready to compute probabilities
            % We want the probability of a response in each response time
            % bin. Not all stimuli contribute to the probability at each
            % time bin, fortunately, so we get some efficiency by
            % identifying those stimuli that do contribute, and only use
            % those in computations for that time bin. Also, the set of
            % stimuli that do contribute is fixed for the duration between
            % consecutive stimuli, so we don't have to iterate over each
            % time bin; instead we iterate over each stimulus and do a
            % vectorized computation on that time chunk.
            
            lp = cell(numel(stim_idx), 1);
            [istart, istop] = deal(zeros(numel(stim_idx), 1));
            for i = 1:numel(stim_idx) % Making this parfor doubles proc time.
                [lp{i}, istart(i), istop(i)] = ...
                    obj.pChunk(i, stim_idx, rra, pdf);
            end
            p = zeros(1, num_t);
            for i = 1:numel(lp)
                p(istart(i):istop(i)) = lp{i};
            end
        end
    end
    methods (Static)
        function [p, idx_start, idx_stop] = pChunk(idx, st, rra, pdf)
            
            pdfl = length(pdf);
            
            idx_start = st(idx)+1;
            
            if idx==numel(st)
                idx_stop = idx_start+pdfl;
            else
                idx_stop = min(idx_start+pdfl, st(idx+1));
            end
            
            candstimidx = (st>(idx_start-pdfl) & st<=idx_start);
            if ~any(candstimidx)
                p = zeros(1, (idx_stop-idx_start));
                return;
            end
            
            cstims = st(candstimidx);
            rr = rra(candstimidx);
            rowp = zeros(numel(cstims), idx_stop-idx_start+1);
            for i = 1:numel(cstims)
                sprev = cstims(i);
                scurr = st(idx);
                snext = idx_stop;
                pdfs = scurr-sprev+1;
                pdfe = min(snext-sprev, pdfl);
                psnip = pdf(pdfs:pdfe).*rr(i);
                rowp(i, 1:numel(psnip)) = psnip;
            end
            p = 1-prod(1-rowp);
            
        end
    end
    
    %% transforms
    methods (Static)
        function x = logit(p)
            x = log(p./(1-p));
        end
        function p = logistic(x)
            p = 1./(1+exp(-x));
        end
    end
    %% Debugging
    methods
        function plotPriors(obj)
            %% look at these choices...
            t = 0.01:.01:1.5;
            figure('name', 'Prior Visualization');
            subplot(2,1,1);
            hold on;
            ax = gca;
            h(1) = plot(ax, t, obj.pr_mu(log(t)));
            h(2) = plot(ax, t, obj.pr_sig(log(t)));
            h(3) = plot(ax, t, obj.pr_tau(log(t)), '--');
            title(ax, 'RT priors');
            legend(ax, h, {'\mu', '\sigma', '\tau'});
            
            subplot(2,1,2);
            ax = gca;
            x = 0:.01:1;
            plot(ax, x, obj.pr_h(obj.logit(x)));
            title('HR prior');
            
        end
        function plotPosteriors(obj, th)
            
            tstr = {'\mu', '\sigma', '\tau'};
            
            figure('name', 'Posterior Visualization');
            
            subplot(1,4,1);
            histogram(obj.logistic(th(:, 1)));
            title('HR');
            
            for i = 1:3
                subplot(1,4,(i+1))
                histogram(exp(th(:, i+1)));
                title(tstr{i});
            end
            
            
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