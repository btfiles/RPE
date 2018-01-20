classdef RSVPPerformanceML3 < rpe.RSVPPerformanceML2
    methods
        function obj = RSVPPerformanceML3(varargin)
            obj@rpe.RSVPPerformanceML2(varargin{:});
        end
        function p = pRespLocal(obj, t_stim, rr, t_min, t_max)
            % probability of responding in a local piece of an RSVP experiment.
            
            % t_stim = 0:.1:1.5;
            if numel(rr)==1
                rr = rr.*ones(size(t_stim));
            end
            assert(numel(rr)==numel(t_stim), 'Resonse Rate must be either scalar or the same size as t_stim.');

            % Create a matrix of individual probabilities
            idx = 0:(obj.pdf_support/obj.time_resolution-1);
            ind_prob = zeros(numel(t_stim), numel(obj.t), class(rr));

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
            
            % p is one minus the probability that no responses occured at time t, which
            % is easy to compute!
            p = 1 - prod(1-ind_prob);
        end
        
        function p_all = pResp(obj, hr, far, exg)
            p_all = gpu_pResp(hr, far, exg, obj.stimulus_times, obj.stimulus_labels, ...
                obj.t(1), obj.time_resolution, numel(obj.t), obj.pdf_support);
        end
    end
end