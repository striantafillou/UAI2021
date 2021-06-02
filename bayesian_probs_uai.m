function bprobs = bayesian_probs_uai(y, sets, nSets, configs, nConfigs, data, domainCounts)

configs_n = nan(nConfigs, nVars);
configs_n(:, sets(end, :))= configs;
nConfigs = size(configs,1);
bprobs = nan(domainCounts(y), nConfigs, nSets);

for iSet=1:nSets
    curSet = find(sets(iSet,:));
    [curProbs, ~, confs, Nexp{iSet}] = cond_prob_mult_inst(y, curSet, data, domainCounts);    
    nConfs = size(confs, 1);
    for iConfig=1:nConfs
        curConf = confs(iConfig,:);
        inds = ismember(configs_n(:, curSet), curConf, 'rows');
        bprobs(:, inds, iSet) = repmat(curProbs(:, iConfig), 1, sum(inds));
    end
end