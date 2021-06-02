function [truePs] = estimateTruePs(testData, jtalg_do_true, imby_true)

nSamples = size(testData,1);
truePs = nan(nSamples, 2);
for iSample =1:nSamples
    truePs(iSample, :) = jtalg_do_true.getConditionalProbabilities(1, imby_true-1, testData(iSample, imby_true));
end
truePs = truePs( :,1);
end