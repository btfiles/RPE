% Vaguely informative priors from Farrell, S., & Ludwig, C. J. (2008).
% Bayesian and maximum likelihood estimation of hierarchical response time
% models. Psychonomic bulletin & review, 15(6), 1209-1217.


x = linspace(0, 3, 10000);

gam1 = 0.2;
gam2 =2.6;
delt1 = 2;
delt2 = .1;
eps1 = 0.1;
eps2 = 0.001;


%%

mu = @(x) normpdf(x, gam1, 1./gam2);
sig = @(x) gampdf(x, delt1, delt2);
tau = @(x) gampdf(x, eps1, eps2);


figure;
hold on;
% plot(x, mu(x));
plot(x, sig(x), '-');
% plot(x, tau(x), '--');