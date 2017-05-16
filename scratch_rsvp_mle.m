%% Response likelihood
%
% For behavioral analyses of RSVP target detection, we want to estimate the
% HR and FAR for a subject. Amar & my regression method does this, but it
% has some trouble when either FAR or HR is close to zero and/or 1.
%
% An alternative would be maximum likelihood estimation, but for that we
% need an estimate of the probability of obtaining a particular result
% under a given set of model parameters (where the model parameters are HR
% and FAR).
%
% Here, I'm trying to figure out the math for computing the probability of
% a given realization of the data given those parameters.
%
% I have a way to figure the probability that a single stimulus will
% generate a response at a particular time, but I'm struggling with how to
% deal with multiple stimuli that might generate a response at a particular
% time.
%
% The usual way to deal with independent events is to sum them, but then I
% end up with probabilities greater than 1.  That's a bummer. It turns out
% they're not really independent, because you can't have multiple responses
% at once (they get collapsed to one).
%
% Let's start with a one-stimulus case.

%% general settings
t_support = 1.5;
t_step = 0.001;
mu = .3;
sig = .05;
tau = .1;

rtpdf = @(x) exgaussPdf(x, mu, sig, tau);
t_pdf = 0:t_step:t_support;

rr = 1; % response rate
%% one stimulus, all alone
t_stim = 0;

t = 0:t_step:(max(t_stim)+t_support);

figure;
plot(t, rr*rtpdf(t_pdf).*t_step);

%% calls two stimuli on the phone.
t_stim = [0 0.25];

t = 0:t_step:(max(t_stim)+t_support);
ind_prob = zeros(numel(t_stim), numel(t));
for iStim = 1:numel(t_stim)
    start_idx = (t_stim(iStim)/t_step)+1;
    ind_prob(iStim, start_idx + (0:(t_support/t_step))) = rtpdf(t_pdf).*t_step*rr;
end
figure;
subplot(2,1,1);
plot(t,ind_prob);
ylabel('individual probability');

subplot(2,1,2);
hold on;
plot(t, sum(ind_prob));
plot(t, prod(ind_prob));
plot(t, sum(ind_prob)-prod(ind_prob));
legend('sum', 'product', 'sum-prod');

%% Let's try 3
t_stim = [0 0.25 0.50];

t = 0:t_step:(max(t_stim)+t_support);
ind_prob = zeros(numel(t_stim), numel(t));
for iStim = 1:numel(t_stim)
    start_idx = (t_stim(iStim)/t_step)+1;
    ind_prob(iStim, start_idx + (0:(t_support/t_step))) = rtpdf(t_pdf).*t_step*rr;
end

figure;
plot(t, ind_prob);

jp = sum(ind_prob,1);
plot(t, jp);
%% the products get trickier.
nprod = sum(arrayfun(@(k) nchoosek(numel(t_stim), k), 2:numel(t_stim)));
prods = zeros(nprod, numel(t));
prod_idx = 1;

for k = 2:numel(t_stim)
    sel_mat = nchoosek(1:numel(t_stim), k);
    for i = 1:size(sel_mat, 1)
        prods(prod_idx, :) = prod(ind_prob(sel_mat(i,:), :), 1);
        prod_idx = prod_idx + 1;
    end
end
figure;
plot(t, sum(ind_prob)-sum(prods));

%% and 4
t_stim = [0 0.25 0.50];

t = 0:t_step:(max(t_stim)+t_support);
ind_prob = zeros(numel(t_stim), numel(t));
for iStim = 1:numel(t_stim)
    start_idx = (t_stim(iStim)/t_step)+1;
    ind_prob(iStim, start_idx + (0:(t_support/t_step))) = rtpdf(t_pdf).*t_step*rr;
end

figure;
subplot(3,1,1);
plot(t, ind_prob);


nprod = sum(arrayfun(@(k) nchoosek(numel(t_stim), k), 2:numel(t_stim)));
prods = zeros(nprod, numel(t));
prod_idx = 1;

for k = 2:numel(t_stim)
    sel_mat = nchoosek(1:numel(t_stim), k);
    for i = 1:size(sel_mat, 1)
        prods(prod_idx, :) = prod(ind_prob(sel_mat(i,:), :), 1);
        prod_idx = prod_idx + 1;
    end
end

subplot(3,1,2)
plot(t, prods);
ylabel('products');

subplot(3,1,3);
plot(t, sum(ind_prob)-sum(prods));

%% Let's go crazy
t_stim = 0:.1:1.5;

t = 0:t_step:(max(t_stim)+t_support);
ind_prob = zeros(numel(t_stim), numel(t));
for iStim = 1:numel(t_stim)
    start_idx = round((t_stim(iStim)/t_step)+1);
    ind_prob(iStim, start_idx + (0:(t_support/t_step))) = rtpdf(t_pdf).*t_step*rr;
end

k_max_list = 2:numel(t_stim);
all_prod = zeros(numel(k_max_list), numel(t));
for iMaxK = 1:numel(k_max_list)
    k_max = k_max_list(iMaxK);
    k_list = 2:k_max;
    nprods = arrayfun(@(k) nchoosek(numel(t_stim), k), k_list);
    nprod = sum(nprods);
    
    k_idx = 0;
    prodc = cell(numel(k_list), 1);
    tic
    parfor i = 1:numel(k_list)
        k = k_list(i);
        sel_mat = nchoosek(1:numel(t_stim), k);
        prods = zeros(nprods(i), numel(t));
        for j = 1:size(sel_mat, 1)
            prods(j, :) = prod(ind_prob(sel_mat(j,:), :), 1); %#ok
        end
        prodc{i} = prods;
    end
    toc
    all_prod(iMaxK, :) = sum(cat(1, prodc{:}));
end
%%
figure; 
plot(t, all_prod);

