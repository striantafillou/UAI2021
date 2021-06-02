function [score,score_in] = dirichlet_bayesian_score(counts, priors)
% dirichlet_bayesian_score calculates 
% log \int \prod_y P(D|theta_y) f(theta_i) d(theta_i)
% counts is the times Y=i [domainCounts(y)\times nConfigs vector]
% prior is an optional multidimensional array of the same shape as counts.
% It defaults to a uniform prior.
% Calculated as
% https://stephentu.github.io/writeups/dirichlet-conjugate-prior.pdf, page
% 6.

if nargin==1
    priors = zeros(size(counts));
end
N= sum(counts);
score_in = gammaln(sum(priors+1))-sum(gammaln(priors+1))+sum(gammaln(counts+priors+1))-gammaln(N+sum(priors+1));
%gammaln(dc)+sum(gammaln(counts+priors+1))- gammaln(sum(counts+priors+1));
score = sum(score_in,2);
end

