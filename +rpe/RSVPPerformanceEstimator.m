classdef RSVPPerformanceEstimator < handle
    % An implementation of a regression-based method for estimating hit
    % rate and false alarm rate in an RSVP target detection experiment.
    %
    % Justification, derivation, and simulations validating this method are
    % presented in Files & Marathe, 2016.
    %
    % Methods will attempt to use parallel for (parfor) loops if the
    % parallel processing toolbox function gcp() is available and returns
    % without error.
    %
    % Example
    % e = rpe.RSVPPerformanceEstimator(stim_time, stim_lbl, button_time);
    % [hr, far] = e.runEstimates();
    %
    % Reference
    % Files, B. T., & Marathe, A. R. (2016). A regression method for
    % estimating performance in a rapid serial visual presentation target
    % detection task. Journal of Neuroscience Methods, 258, 114?123.
    % http://doi.org/10.1016/j.jneumeth.2015.11.003
    %
    % RSVPPerformanceEstimator properties:
    %   Must be set by user (in constructor)
    %   stimulus_times - times (s) at which stimuli were presented
    %   stimulus_labels - true for targets, false otherwise
    %   buttonpress_times - times (s) at which button was pressed
    %
    %   Options with default values
    %   time_resolution - resolution of PDF approximation (s)
    %                       default .001.
    %   response_window - response time window for RT estimates (s)
    %                       default [0.0 1.0].
    %   pdf_support - how long after the stimulus to compute RT PDF (s)
    %                       default 1.5
    %
    % RSVPPerformanceEstimator methods:
    %   RSVPPerformanceEstimator - Constructor takes 3 arguments:
    %                               stim_times, stim_lbls, button_times
    %   runEstimates - Estimates the response time PDF and uses that to
    %                   estimate HR and FAR.  Uses estimatePdf and
    %                   estimatePerfromance
    %
    % See Also example_script
    
    
    properties
        stimulus_times; % times (s) at which stimuli were presented
        stimulus_labels; % true for targets, false otherwise
        buttonpress_times; % times (s) at which button was pressed
        
        % configurable parameters
        
        time_resolution = 0.001; % resolution of PDF approximation
        response_window = [0.0 1.0]; % response time window for RT estimates
        pdf_support = 1.5; % how long after the stimulus to compute RT PDF
    end
    
    properties (GetAccess=public, SetAccess=protected)
        % estimated parameters of the exGaussian response time distribution
        mu % mean of the gaussian
        sigma % standard deviation of the gaussian
        tau % parameter of the exponential
        
        % best guess at collection of response times
        rt_list
        
        % rt pdf convenience values:
        
        pdf_fcn % function handle for the PDF estimate
        pdf_est % pre-computed PDF values
        
        beta            % explanatory variable for regression
        response_scores % dependent variable for regression
    end
    
    methods
        function obj = RSVPPerformanceEstimator(varargin)
            % Takes 3 arguments: stim_time, stim_lbl, buttonpress_time.
            if nargin == 0,
                return
            end
            
            ip = inputParser();
            ip.addRequired('stim_t');
            ip.addRequired('stim_lbl');
            ip.addRequired('bp_t');
            
            ip.parse(varargin{:});
            obj.stimulus_times = ip.Results.stim_t;
            obj.stimulus_labels = ip.Results.stim_lbl;
            obj.buttonpress_times = ip.Results.bp_t;
        end
        function [hr, far, hrci, farci] = runEstimates(obj, cialpha)
            % Estimate the rt pdf, HR and FAR.
            % [hr, far] = updateEstimates()
            %
            % See also estimatePdf estimatePerformance
            
            if nargin < 2,
                cialpha = .05;
            end
            
            obj.estimatePdf;
            if nargout == 2,
                [hr, far] = obj.estimatePerformance;
            elseif nargout == 4,
                [hr, far, hrci, farci] = obj.estimatePerformance(cialpha);
            end
        end
        
        function estimatePdf(obj)
            % estimates a new response time PDF
            % Stimulus labels and stimulus and button press times must
            % already be set.
            % See also rpe.fitExGauss rpe.exGaussPdf
            
            assert(~isempty(obj.stimulus_times), ...
                'RPE:TrainPdf:MissingStimulusTimes', ...
                'Cannot train response PDF with no stimuli.');
            assert(~isempty(obj.stimulus_labels), ...
                'RPE:TrainPdf:MissingStimulusLabels', ...
                'Cannot train response PDF with no stimulus labels.');
            assert(~isempty(obj.buttonpress_times), ...
                'RPE:TrainPdf:MissingButtonPress', ...
                'Cannot train response PDF with no button presses.');
            
            %% build the RT list
            obj.buildRTList();
            
            %% from that vector, fit an exGaussian
            % ex gaussian is a distribution arising from a random variable
            % that is the sum of a normally distributed random variable and
            % an exponentially distributed random variable.  The exGaussian
            % has three parameters: tau is the parameter of the exponential
            % distribution and mu & sigma are the mean and standard
            % deviation of the normal distribution.
            %
            % These values are fit using maximum likelihood estimation.
            [obj.mu,obj.sigma,obj.tau] = rpe.fitExGauss(obj.rt_list);
            obj.pdf_fcn = @(rt) rpe.exGaussPdf(rt, ...
                obj.mu,obj.sigma,obj.tau);
            
            %% build a density approximation at requested resolution
            t = obj.time_resolution:obj.time_resolution:obj.pdf_support;
            obj.pdf_est = obj.pdf_fcn(t);
        end
        function [HR, FAR, HRCI, FARCI] = estimatePerformance(obj, alph)
            % Estimate HR and FAR
            
            % Build Beta
            obj.buildBeta;
            
            % Build Response Scores
            obj.buildResponseScores;
            
            % Solve for HR and FAR
            %o = obj.beta\[obj.response_scores]';
            if nargout == 2,
                o = regress(obj.response_scores', obj.beta);
            else
                if nargin < 2,
                    alph = .05;
                end
                [o, ci] = regress(obj.response_scores', obj.beta, alph);
                HRCI = ci(1,:);
                FARCI = ci(2,:);
            end
            
            HR = o(1);
            FAR = o(2);
            
            % Correct HR/FAR
            if HR > 1,
                HR = 1;
            elseif HR < 0,
                HR = 0;
            end
            if FAR > 1,
                FAR = 1;
            elseif FAR < 0,
                FAR = 0;
            end
        end
    end
    methods (Access=protected)
        function buildRTList(obj)
            % To estimate the rt pdf, we need a collection of RTs.  Because
            % we don't know what stimuli evoked which responses, some
            % heuristic is needed. The collection of RTs is built using the
            % window method. This means we go over each target and look at
            % a window of time after that target.  The first response that
            % happens in that window is assumed to be evoked by that
            % target, so their difference is added to a collection of RTs.
            %
            % This will, of course, be wrong sometimes, but it's the best
            % we can do (that I can think of).
            
            %% Initialize the parallel pool, if possible/needed
            try
                gcp();
            catch
                % no parallel toolbox
            end
            %% find stimuli labeled as targets
            tar_times = obj.stimulus_times(obj.stimulus_labels==true);
            %% build a vector of response times
            rts = zeros(size(tar_times));
            bpt = obj.buttonpress_times;
            rw = obj.response_window;
            % Note: creating local versions of these variables allows
            % faster parallel execution.
            
            parfor iTar = 1:numel(tar_times),
                tt = tar_times(iTar);
                resp_idx = find(bpt < tt+rw(2) & bpt > tt+rw(1)); %#ok
                if numel(resp_idx) == 0,
                    %miss.
                    continue
                elseif numel(resp_idx)>1,
%                     warning('RPE:BuildRTDist:MultiResponse',...
%                         ['The stimulus at time %f is followed by more ',...
%                         'than one responses (%d). Taking only the ',...
%                         'first.'],tt,numel(resp_idx));
                    resp_idx = resp_idx(1);
                end
                rts(iTar) = bpt(resp_idx)-tt;
            end
            rts = rts(rts~=0);
            obj.rt_list = rts;
        end
        function buildBeta(obj)
            %Assemble the regression coefficients
            try
                gcp();
            catch
                %parallel not available
            end
            l_beta = zeros(numel(obj.stimulus_times),2);
            
            mxrt = obj.pdf_support;
            ost = obj.stimulus_times;
            osl = obj.stimulus_labels;
            tr = obj.time_resolution;
            l_pdf_fcn = obj.pdf_fcn;
            l_pdf_support = obj.pdf_support;
            
            parfor iStim = 1:numel(ost),
                % get si, the stim of interest and sj, the list of stimuli
                % whos responses could be attributed to si
                si = ost(iStim);
                idx_neighbor = ost >= (si-mxrt) & ...
                    ost <= (si+mxrt);
                neighbor_times = ost(idx_neighbor);
                neighbor_labels = osl(idx_neighbor); %#ok
                
                % compute expected attribution from each of sj, partitioned
                % as hit contributions and fa contributions.
                b1 = 0;
                b2 = 0;
                for jNeighbor = 1:numel(neighbor_times),
                    sj = neighbor_times(jNeighbor);
                    % figure out times of responses that could be generated
                    % by sj and could contribute to attribution of si
                    t_min = max(si, sj);
                    t_max = min(si+mxrt, sj+mxrt);
                    t = t_min:tr:t_max;
                    
                    % now compute attribution for each t
                    a = att(t,si, l_pdf_fcn, ost, l_pdf_support); %#ok
                    
                    % compute the contribution to si of a response
                    % by sj conditioned on a response by sj.
                    e = sum(tr.*l_pdf_fcn(t-sj).*a);
                    
                    lbl = neighbor_labels(jNeighbor);
                    if lbl,
                        b1 = b1+e;
                    else
                        b2 = b2+e;
                    end
                end
                l_beta(iStim,:) = [b1 b2];
            end
            obj.beta = l_beta;
        end
        
        function buildResponseScores(obj)
            % Attribute each response to possible evoking stimuli.
            obj.response_scores = zeros(size(obj.stimulus_times));
            for iResp = 1:numel(obj.buttonpress_times),
                t = obj.buttonpress_times(iResp);
                candidate_idx = obj.stimulus_times < t & ...
                    obj.stimulus_times > t - obj.pdf_support;
                
                if ~any(candidate_idx),
                    % Rogue buttonpress.
                    continue;
                end
                
                st = t-obj.stimulus_times(candidate_idx);
                lik = obj.pdf_fcn(st);
                scores = lik./sum(lik);
                obj.response_scores(candidate_idx) = scores + ...
                    obj.response_scores(candidate_idx);
            end
        end
    end
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