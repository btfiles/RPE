function L = rsvpLikelihood(stim_time, stim_label, button_time, t_res, t_support, hr, exgth)
% Compute the likelihood of observing results indicated in button time for
% the rsvp experiment described by stim_time and stim_label, given a hit
% rate HR and an RT distrubtion that is ex-gaussian with parameters theta
% having support from 0 to t_support.

% determine far
nresp = numel(button_time);
nTar = sum(stim_label==1);
nNt = sum(stim_label==0);
%nresp = hr*nTar + far*nNt;
far = (nresp-hr*nTar)/nNt;

% figure out time
t_min = min(stim_time);
t_max = max(max(stim_time), max(button_time))+t_support;

T = t_min:t_res:t_max;
p = zeros(size(T));
for i = 1:numel(T)
    t = T(i);
    
    local_stim_idx = stim_time >= t-t_support & stim_time < t;
    
    local_tar_delt = t-stim_time(local_stim_idx & stim_label==1);
    local_nontar_delt = t-stim_time(local_stim_idx & stim_label==0);
    
    pNoHit = prod(1-hr.*rpe.exGaussPdf(local_tar_delt, exgth(1), exgth(2), exgth(3)).*t_res);
    pNoFa = prod(1-far.*rpe.exGaussPdf(local_nontar_delt, exgth(1), exgth(2), exgth(3)).*t_res);
    
    p(i) = 1-pNoHit*pNoFa;
end

figure;
plot(T, p);
L=[];


end