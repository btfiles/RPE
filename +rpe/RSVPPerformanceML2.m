classdef RSVPPerformanceML2 < rpe.RSVPPerformanceML
    % Use bounded maximum likelihood estimation rather than regression to
    % determine HR and FAR as well as the parameters of the exgaussian.
    %
    % 5/16/2017 BTF
    
    properties
        hr_init = 0.9;
        far_init = 0.01;
        new_opt = optimset('Display', 'iter', 'MaxIter', 400, ...
            'TolFun', 0.1, 'TolX', 1e-5); % Options for fminsearch.
    end
    
    methods
        function obj = RSVPPerformanceML2(varargin)
            obj@rpe.RSVPPerformanceML(varargin{:})
            obj.opt = obj.new_opt;
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
            fcn = @(o) -logLikelihood(obj, o(1), o(2), o(3:5), bp);
            
            initial_values = [obj.hr_init; obj.far_init; ...
                obj.mu; obj.sigma; obj.tau];
            lower_bounds = [0; 0; repmat(obj.time_resolution, 3, 1)];
            upper_bounds = [1; 1; inf(3,1)];
            o = rpe.fminsearchbnd(fcn, initial_values, lower_bounds, ...
                upper_bounds, obj.opt);

            HR = o(1);
            FAR = o(2);
            
            % set the exgaussian parameters to the mle estimate
            exg = o(3:5);
            obj.setPdf(exg);
        end
        
        function llik = logLikelihood(obj, hr, far, exg, bp)
            % bp is a boolean array that is true for time bins with a
            % button press.
            p_resp = pResp(obj, hr, far, exg);
            
            % doing this without the log results in numerical underflow.
            llik = sum(log(p_resp(bp))) + sum(log(1-p_resp(~bp)));
        end
        function setPdf(obj, exg)
            % setup the rt pdf
            obj.mu = exg(1); % mean of the gaussian
            obj.sigma = exg(2); % standard deviation of the gaussian
            obj.tau = exg(3); % parameter of the exponential
            
            obj.pdf_fcn = @(rt) rpe.exGaussPdf(rt, ...
                obj.mu,obj.sigma,obj.tau);
            
            t = obj.time_resolution:obj.time_resolution:obj.pdf_support;
            obj.pdf_est = obj.pdf_fcn(t);
        end
        function p_all = pResp(obj, hr, far, exg)
            % Computes the probability of responding in each time bin given
            % HR and FAR.
            % p_all = pResp(obj, hr, far, exg)
            %
            % Calculates the joint probability of one or more responses
            % occurring in each time bin. An object property, kmax, defines
            % the highest-order interaction this computation takes into
            % account. This determines how many stimuli generating a
            % response at a given time bin will be taken into account. If
            % kmax is set to 3, for example, then the possibility that any
            % combination of 3 (but not 4 or more) of however many nearby
            % stimuli might produce a response ocurred simultaneously.
            %
            % Sets the rt-pdf to an exgaussian with parameters defined in
            % exg.19
            
            obj.setPdf(exg);
            
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

    end
end