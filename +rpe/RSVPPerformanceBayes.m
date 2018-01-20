classdef RSVPPerformanceBayes < rpe.RSVPPerformanceML3
    
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
        function p = prior(obj, theta)
            p = prod([...
                obj.pr_h(theta(1)), ...
                obj.pr_mu(theta(2)), ...
                obj.pr_sig(theta(3)), ...
                obj.pr_tau(theta(4))]);
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
            obj@rpe.RSVPPerformanceML3(varargin{:});
        end
        function [HR, FAR] = estimatePerformance(obj)
            % Estimates performance on a RSVP target detection experiment.
            % [HR, FAR] = estimatePerformance(obj)
            %
            % Uses maximum likelihood estimation to select the HR and FAR
            % that result in the highest probability of obtained results.
            %
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
            
            lik = @(theta) obj.likelihood(theta(1), theta(2:end), bp);
            pr = @(theta) obj.prior(theta);
            for i = 1:100
                thminit = [obj.prs_h, obj.prs_mu, obj.prs_sig, obj.prs_tau];
                if lik(thminit)~=0
                    break
                end
            end
            if i==100
                error('could not find a starting spot with lik~=0.');
            end
            mwg = rpe.MWG(ones(1,4).*.2, lik, pr, thminit);
            
            mwg.nmh = 500;
            mwg.nbi = 50;
            mwg.nbatch = 25;
            
            t0 = tic;
            [theta, thlik, thprior] = mwg.sample();
            obj.trace = theta;
            obj.trace_lik = thlik;
            obj.trace_prior = thprior;
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