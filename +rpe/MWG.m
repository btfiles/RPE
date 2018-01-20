classdef MWG < handle
    % Implements metropolis-within-gibbs sampling
    
    properties
        nmh = 5000; % number of samples
        nbi = 250;  % number of burn-in samples to discard
        
        minadj = 0.01; % minimum adjustment size on sd of proposal distribution, q
        nbatch = 50;   % number of samples between updates on q sd
        target_rej = 0.44; % target proportion of rejected proposals
    end
    
    properties
        % initialize these at instantiation
        qsdinit % initial proposal distribution sd
        qminit  % mean of initial proposal distribution
        dim     % number of dimensions on theta
        lfcn    % likelihood function
        pfcn    % prior function
    end
    
    properties
        qsd
        rej
    end
    
    methods
        function obj = MWG(initial_proposal_sd, likelihood_fcn, prior_fcn, initial_proposal_m)
            % obj = MWG(initial_proposal_sd, likelihood_fcn, prior_fcn)
            %
            
            if nargin==0
                return;
            end
            
            if nargin<3
                error('Three arguments are required: initial_proposal_sd, likelihood_fcn, prior_fcn.');
            end
            
            obj.qsdinit = initial_proposal_sd;
            obj.dim = numel(initial_proposal_sd);
            obj.lfcn = likelihood_fcn;
            obj.pfcn = prior_fcn;
            
            if nargin >= 4
                obj.qminit = initial_proposal_m;
            else
                obj.qminit = zeros(size(obj.qsdinit));
            end
            
        end
        function p = prior(obj, theta)
            p = obj.pfcn(obj.redim(theta));
        end
        function l = likelihood(obj, theta)
            l = obj.lfcn(obj.redim(theta));
        end
        function r = qrnd(obj, idx)
            % A random draw from the proposal distribution (at 0)
            r = normrnd(0, obj.qsd(idx));
        end
        function p = qpdf(obj, theta, idx)
            % Proposal density of theta (relative to 0)
            p = normpdf(theta(idx), 0, obj.qsd(idx));
        end
        
        function th = redim(~, th)
            % helper function to get thetas in line
            sz = size(th);
            if sum(sz~=1)>1
                error('Input parameter vector may only have 1 non-singleton dimension.');
            end
            th = th(:)';
        end
        
        function [p, lik2, prior2] = alph(obj, theta1, theta2)
            % Probability of accepting a move from theta1 to theta2
            % also returns likelihood and prior of theta2, to avoid
            % re-computing
            
            theta1 = obj.redim(theta1);
            theta2 = obj.redim(theta2);
            
            lik2 = obj.likelihood(theta2);
            prior2 = obj.prior(theta2);
            
            numerator = lik2*prior2;
            denominator = obj.likelihood(theta1)*obj.prior(theta1);
            
            if denominator==0
                p = 1;
            else
                p = min(1, numerator./denominator);
            end
        end
        
        function [thg, lik_thg, prior_thg, thj] = sample(obj)
            % Draws samples from posterior distribution using
            % metropolis-within-gibbs.
            % [thg, like_thg, prior_thg, thj] = sample(obj)
            % thg contains the samples, with their associated likelihood
            % and prior values. thj is the proposal increment distribution.
            
            ns = obj.nmh+obj.nbi;
            nd = obj.dim;
            
            % setup storage
            thg = zeros(ns, nd);
            thj = zeros(ns, nd);
            lik_thg = zeros(ns, nd);
            prior_thg = zeros(ns, nd);
            obj.rej = false(ns, nd);
            
            % set initial values
            obj.qsd = obj.qsdinit;
%             thg(1, :) = reshape(arrayfun(@(m, sd) normrnd(0, sd), obj.qminit, obj.qsd), ...
%                 1,[]); % Take an initial draw
            thg(1,:) = obj.qminit;
            lik_thg(1, :) = repmat(obj.likelihood(thg(1,:)), 1, nd);
            prior_thg(1, :) = repmat(obj.prior(thg(1,:)), 1, nd);
            
            % start iterating
            for i = 2:ns
                % update qsd; this is "Metropolis-within-gibbs"
                if mod(i, obj.nbatch) == 0
                    idx = (i-obj.nbatch+1):i;
                    brej = mean(obj.rej(idx,:));
                    adjs = (brej<obj.target_rej)*2-1;
                    adj = min([obj.minadj, i^-.5]);
                    obj.qsd = obj.qsd + adj.*adjs;
                end
                
                for j = 1:nd
                   
                    
                    prev_thg = [thg(i, 1:(j-1)) thg(i-1, j:end)];
                    
                    if j==1
                        prj=nd;
                        pri=i-1;
                    else
                        prj = j-1;
                        pri = i;
                    end
                    
                    prev_lik = lik_thg(pri, prj);
                    prev_prior = prior_thg(pri, prj);
                    
                    thj(i, j) = obj.qrnd(j);
                    
                    prop_thg = prev_thg;
                    prop_thg(j) = prop_thg(j)+thj(i,j);
                    
                    [prop_alph, prop_lik, prop_prior] = obj.alph(prev_thg, prop_thg);
                    
                    if rand() < prop_alph
                        % make the move
                        thg(i, j) = prop_thg(j);
                        lik_thg(i, j) = prop_lik;
                        prior_thg(i, j) = prop_prior;
                    else
                        thg(i, j) = prev_thg(j);
                        lik_thg(i, j) = prev_lik;
                        prior_thg(i, j) = prev_prior;
                        obj.rej(i, j) = true;
                    end
                end
            end
            [thg, lik_thg, prior_thg, thj] = obj.burn(thg, lik_thg, prior_thg, thj);
        end % function
        function varargout = burn(obj, varargin)
            idx = (obj.nbi+1):(obj.nbi+obj.nmh);
            varargout = cell(size(varargin));
            for i = 1:numel(varargin)
                d = varargin{i};
                nd = ndims(d);
                arg = [idx repmat({':'}, 1, nd-1)];
                varargout{i} = d(arg{:});
            end
        end
    end % methods
end % classdef