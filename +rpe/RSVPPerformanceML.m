classdef RSVPPerformanceML < rpe.RSVPPerformanceEstimator
    % Use bounded maximum likelihood estimation rather than regression to
    % determine HR and FAR.
    %
    % 5/3/2017 BTF
    
    properties
        kmax = 5; % highest order interaction to compute
        t;
        
        opt = optimset('Display', 'iter', 'MaxIter', 50, ...
            'TolFun', 1); % Options for fminsearch.
    end
    
    methods
        function obj = RSVPPerformanceML(varargin)
            obj@rpe.RSVPPerformanceEstimator(varargin{:})
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
            % and their labels as fixed. The parameters of the response
            % time distribution are also fixed after estimating them from
            % the results using a heuristic to build a response time
            % sample.
            %
            % It might be more formally correct to treat the parameters of
            % the response time distribution, but that seems way too
            % complicated and/or computationally intensive.
            %
            % Uses fminsearchbnd from matlab file exchange.
            
            % Work out time bins of width time_resolution (a property of
            % this class) and place responses into their bins
            t_min = min([min(obj.stimulus_times), min(obj.buttonpress_times)]);
            t_max = max([max(obj.stimulus_times), max(obj.buttonpress_times)]) + obj.pdf_support;
            
            obj.t = t_min:obj.time_resolution:t_max;
            
            bpidx = floor((obj.buttonpress_times-t_min)/obj.time_resolution) + 1;
            bp = false(size(obj.t));
            bp(bpidx) = true;
            
            % We have a minimizing optimizer, so to get the maximum
            % likelihood estimate, we use -log likelihood. Log likelihood
            % is easier to compute than likelihood, but yields equivalent
            % solutions.
            fcn = @(o) -logLikelihood(obj, o(1), o(2), bp);
            o = rpe.fminsearchbnd(fcn, [1; 0], zeros(2,1), ones(2,1), obj.opt);

            HR = o(1);
            FAR = o(2);
        end
        
        function llik = logLikelihood(obj, hr, far, bp)
            % bp is a boolean array that is true for time bins with a
            % button press.
            p_resp = pResp(obj, hr, far);
            
            % doing this without the log results in numerical underflow.
            llik = sum(log(p_resp(bp))) + sum(log(1-p_resp(~bp)));
        end
        
        function p_all = pResp(obj, hr, far)
            % Computes the probability of responding in each time bin given
            % HR and FAR.
            % p_all = pResp(obj, hr, far)
            %
            % Calculates the joint probability of one or more responses
            % occurring in each time bin. An object property, kmax, defines
            % the highest-order interaction this computation takes into
            % account. This determines how many stimuli generating a
            % response at a given time bin will be taken into account. If
            % kmax is set to 3, for example, then the possibility that any
            % combination of 3 (but not 4 or more) of however many nearby
            % stimuli might produce a response ocurred simultaneously.
            
            % response rate array: HR for targets, FAR for non-targets
            rra = far*ones(size(obj.stimulus_labels));
            rra(obj.stimulus_labels==true) = hr;
            
            %% put stimuli in the same time bins as the responses.
            stim_idx = round(obj.stimulus_times/obj.time_resolution) + 1;
            
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
            
            [p_idx, p_allc] = deal(cell(size(obj.stimulus_times)));
            parfor iStim = 1:numel(obj.stimulus_times) % candidate for parallelization?
                
                % Here we select those stimuli that might contribute to the
                % probability in the time between the current and the next
                % stimulus
                in_hood = (obj.stimulus_times >= (obj.stimulus_times(iStim) - obj.pdf_support)) & ... 
                    (obj.stimulus_times <= obj.stimulus_times(iStim)); %#ok
                
                hood_idx = find(in_hood);
                t_stim_local = obj.stimulus_times(hood_idx) - obj.stimulus_times(hood_idx(1)); % set first one to zero
                
                % and now we need to figure out what part to keep
                if iStim == numel(obj.stimulus_times)
                    t_next = obj.stimulus_times(end)+obj.pdf_support;
                else
                    t_next = obj.stimulus_times(iStim+1);
                end
                
                % rebase times to first stimulus in the neighborhood
                t_min = obj.stimulus_times(iStim) - obj.stimulus_times(hood_idx(1));
                t_max = min([t_next - obj.stimulus_times(hood_idx(1)), ...
                    obj.stimulus_times(iStim) + obj.pdf_support - obj.stimulus_times(hood_idx(1))]);
                
                % get the bit of the probability timeline
                p_local = pRespLocal(obj, t_stim_local, rra(in_hood), t_min, t_max); %#ok
                
                % put the result in a cell array for later use (we have to
                % do this to keep parfor happy)
                p_idx{iStim} = stim_idx(iStim) - 1 + (1:numel(p_local));
                p_allc{iStim} = p_local;
            end
            
            % Iterate over the results and compute the full timeline of
            % probabilities.
            p_all = zeros(size(obj.t));
            for iStim = 1:numel(p_idx)
                p_all(p_idx{iStim}) = p_allc{iStim};
            end
        end
        function p = pRespLocal(obj, t_stim, rr, t_min, t_max)
            % probability of responding in a local piece of an RSVP experiment.
            
            % t_stim = 0:.1:1.5;
            if numel(rr)==1
                rr = rr.*ones(size(t_stim));
            end
            assert(numel(rr)==numel(t_stim), 'Resonse Rate must be either scalar or the same size as t_stim.');
            
            if obj.kmax > 14
                warning('Selecting a high value of kmax can cause this algorithm to be very slow. Consider kmax < 14.');
            end
            if numel(t_stim)~=1 && obj.kmax > numel(t_stim)
                obj.kmax = numel(t_stim);
            end
            
            % Create a matrix of individual probabilities
            idx = 0:(obj.pdf_support/obj.time_resolution-1);
            ind_prob = zeros(numel(t_stim), numel(obj.t));
            
            start_idx = round((t_stim/obj.time_resolution)+1);
            for iStim = 1:numel(t_stim)
                ind_prob(iStim, start_idx(iStim) + idx) = obj.pdf_est*obj.time_resolution*rr(iStim);
            end
            
            % cull out just the part we care about
            cull_start = floor(t_min/obj.time_resolution)+1;
            cull_end = floor(t_max/obj.time_resolution)+1;
            
            ind_prob = ind_prob(:, cull_start:cull_end);
            
            if numel(t_stim)==1
                % no interactions to consider
                p = ind_prob;
                return;
            end
            
            % compute interaction terms of order 2 - kmax.
            k_list = 2:obj.kmax;
            nprods = arrayfun(@(k) nchoosek(numel(t_stim), k), k_list);
            prodc = cell(numel(k_list), 1);
            % tic
            for i = 1:numel(k_list)
                k = k_list(i);
                sel_mat = nchoosek(1:numel(t_stim), k);
                prods = zeros(nprods(i), size(ind_prob, 2));
                for j = 1:size(sel_mat, 1)
                    prods(j, :) = prod(ind_prob(sel_mat(j,:), :), 1);
                end
                prodc{i} = prods;
            end
            % toc
            
            % p = sum of individual probabilities minus their co-occurances
            % (i.e. their products)
            p = sum(ind_prob) - sum(cat(1, prodc{:}));
        end
    end
end