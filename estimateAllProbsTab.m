function [bprobs_av, logbprobs] = estimateAllProbsTab(sets, nSets, cmbconfigs, Nexp, Nobs, probsExp, domainCounts, y, nVars, scores)

nConfigs = size(cmbconfigs{nSets}, 1);
logbprobs = zeros(nSets+1, domainCounts(y), nConfigs);
[logbprobs_av, bprobs_av] =deal(zeros(domainCounts(y), nConfigs));
for iSet=1:nSets
    logbprobs(iSet+1, :, :)= reshapeProbs(probsExp{iSet}, sets(iSet,:), sets(nSets, :), cmbconfigs{iSet}, cmbconfigs{nSets}, nVars, domainCounts, y);
end
logbprobs(1, :, :)= reshapeProbs(dirichlet_posterior_expectation(Nexp{nSets}, Nobs{nSets}), sets(iSet,:), sets(nSets, :), cmbconfigs{iSet}, cmbconfigs{nSets}, nVars, domainCounts, y);

for iConfig =1:nConfigs
    for iy =1:domainCounts(y)
        logbprobs_av(iy, iConfig) = sumOfLogsV(scores(nSets:2*nSets)+ log(logbprobs(:, iy, iConfig)));
    end
    bprobs_av(:, iConfig) = exp(logbprobs_av(:, iConfig)-sumOfLogsV(logbprobs_av(:, iConfig)));
end


end     