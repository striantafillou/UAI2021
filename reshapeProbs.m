function bprobs = reshapeProbs(probs, set, bigSet, configs, bigConfigs, nVars, domainCounts, y)

configs_n = nan(size(bigConfigs, 1), nVars);
configs_n(:, bigSet)= bigConfigs;
nConfigs = size(configs_n,1);
bprobs = nan(domainCounts(y), nConfigs);
 
for iConfig=1:size(configs, 1)
    curConf = configs(iConfig,:);
    inds = ismember(configs_n(:, set), curConf, 'rows');
    bprobs(:, inds) = repmat(probs(:, iConfig), 1, sum(inds));
end
end