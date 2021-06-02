ed% test the new algorithm
clear; clc;
comp ='sot16.PITT'; 
%comp ='sofia'
javaaddpath(['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs\tetrad\custom-tetrad-lib-0.2.0.jar'])
import edu.cmu.tetrad.*
import java.util.*
import java.lang.*
import edu.cmu.tetrad.data.*
import edu.cmu.tetrad.search.*
import edu.cmu.tetrad.graph.*
import edu.cmu.tetrad.bayes.*

code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs'];
addpath(genpath(code_path));
code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_effects'];
addpath(genpath(code_path));
cd([code_path ]);

% control panel
N=10000;x=1; y=2;
doNs =[50 100 500  ]; maxDoN = doNs(end);nDoNs =length(doNs);
doInds = getExperimentalInds(maxDoN, doNs, nDoNs, 2);
% simulate data from X->Y, X<-W1->W2->Y
% nVarsOr =5;
% dag = zeros(nVarsOr);
% dag(1, 2)=1; dag(3, 1) =1; dag(3, 4)=1; dag(4, 2)=1;dag(5, 2)=1;
% isLatent = [false false false false false];
%load twoconfs.mat
load mbias_bn.mat  

%load confandemgraph.mat
nVars= sum(~isLatent);
smm = dag2smm(dag, isLatent);
domainCountsOr =[2 2 randi([2 2], 1, nVarsOr-2)];

nIters=100;
%scores = nan(nSets*2, nDoNs, nIters);

% initialize
auc = nan( nIters, 3, nDoNs);
[Xs, Ys] = deal(nan(21, nIters, 3, nDoNs));
[cmbProbs, mbeProbs, mbProbs] = deal(nan(1000, nIters, nDoNs));
trueEvents = nan(2000, nIters);
diffs = nan(nIters,1);

for iter=1:nIters
    fprintf('Iter %d \n', iter);
    %initialize
  
    % nodesOr: with confounders.
   % [nodesOr, domainCountsOr, orderOr, rnodes] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCountsOr, 'minNumStates', domainCountsOr);
   [tIM] = tetradEIM(dag, nodesOr, domainCountsOr);
    jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tIM);

%    mby = findMBsmm(smm,y);mby = setdiff(mby, 1);

   % diffs(iter) =mean(abs(idTruePop(1, :)-idBiased(1, :)));
    % simulate data with dummy nodes.
    obsDatasetOr = simulatedata(nodesOr,N, 'discrete', 'domainCounts', domainCountsOr, 'isLatent', isLatent); % simulate data from original distribution
    % observational data, learn dag, and BN
    obsDataset = subdataset(obsDatasetOr);domainCounts=obsDataset.domainCounts;
    
    [adag, dsj]= tetradFges12(obsDataset.data, domainCounts, 'pretreat', true, 'onlyY', true);
    mby = setdiff(find(adag(:, 2)), 1)';
    
    dTruePop = estimateDoProbCondJT(2,1,mby,dag, nodesOr, domainCountsOr);
    idBiased = estimateCondProbJT(2, [1 mby], jtalg, nVarsOr, domainCountsOr);

    
    [sets, nSets] = allSubsets(nVars,mby);
    sets(:, x) = true;
    [logpDegivDoHw, logpDegivDoHn] =deal(nan(nSets,nDoNs));    
    logpDogivH = nan(nSets,1);
    [curSet, Nexp, Nobs,cmbconfigs, nConfigs,pobs, probsExp] = deal(cell(1, nSets));
    for iSet=1:nSets
        %fprintf('iSet %d-----------\n', iSet)
        curSet{iSet} = find(sets(iSet,:));
        [pobs{iSet}, ~, cmbconfigs{iSet}, Nobs{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], obsDataset.data, domainCounts(1:nVars));
        nConfigs{iSet} = size(cmbconfigs{iSet}, 1);
        logpDogivH(iSet) = dirichlet_bayesian_score(Nobs{iSet});
    end


%     list= tetradList(nVars, domainCounts);
%     % make tetrad data set
%     ds2 = javaObject('edu.cmu.tetrad.data.VerticalIntDataBox',obsDataset.data');
%     dsj = javaObject('edu.cmu.tetrad.data.BoxDataSet',ds2, list);
%     % make algo
%     aGraph=dag2tetrad(adag,list, nVars); % graph
%     tBM= javaObject('edu.cmu.tetrad.bayes.BayesPm', aGraph);
%     eIMprior = edu.cmu.tetrad.bayes.DirichletBayesIm.symmetricDirichletIm(tBM, 1)%     eIMpost= edu.cmu.tetrad.bayes.DirichletEstimator.estimate(eIMprior, dsj);% 

    % simulate experimental data 
    expDsOr = simulateDoData(nodesOr, 1,  0:domainCountsOr(1)-1, maxDoN, 'discrete', 'domainCounts', domainCounts);
    expDs = subdataset(expDsOr,isLatent);
    expData = expDs.data;
    



    testDsOr = simulateDoData(nodesOr, 1,  0:domainCountsOr(1)-1, 1000, 'discrete', 'domainCounts', domainCounts);    
    testDs = subdataset(testDsOr,isLatent);
    testData = testDs.data;

    for iDoN =1:nDoNs
            [adage]= tetradFges12(expData(doInds(:, iDoN), :), domainCounts, 'pretreat', true, 'onlyY', true);
            mbye = find(adage(:, 2))';%setdiff(find(adage(:, 2)), 1)';
            [probsmbye, ~, mbyeconfigs] = cond_prob_mult_inst(y, mbye, expData(doInds(:, iDoN), :), domainCounts(1:nVars));
        
        for iSet=1:nSets
            %curSet{iSet} = find(sets(iSet,:));
            [probsExp{iSet}, ~, ~, Nexp{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], expData(doInds(:, iDoN), :), domainCounts(1:nVars));            
            logpDegivDoHw(iSet, iDoN) = dirichlet_bayesian_score(Nexp{iSet}, Nobs{iSet});
            logpDegivDoHn(iSet, iDoN) = dirichlet_bayesian_score(Nexp{iSet});
        end
        numer = [logpDegivDoHw(:, iDoN)+logpDogivH;logpDegivDoHn(:, iDoN)+logpDogivH];
        denom =sumOfLogsV(numer);
        scores = numer-denom;
        [~, b] =max(scores);
        if b<nSets+1
            bestSet = curSet{b};
            configs = cmbconfigs{b};
            bestprobs = dirichlet_posterior_expectation(Nexp{b}, Nobs{b});
        else
            bestSet = curSet{b-nSets};
            configs = cmbconfigs{b-nSets};
            bestprobs = probsExp{b-nSets};
        end
        [auc(iter,1, iDoN)]=estimateAUCfs(bestSet, bestprobs, configs, testData);
        [auc(iter,2, iDoN)] =estimateAUCfs(mbye, probsmbye, mbyeconfigs, testData);
        [auc(iter,3, iDoN)] =estimateAUCfs(curSet{end},pobs{end},  cmbconfigs{end}, testData);
     

        % against dexp all

    end
    
end
%%

close all;
figure;hold all;
for iDoN=1:nDoNs
    subplot(1, nDoNs,iDoN); ah = gca;boxplot(auc(:,:, iDoN)); hold all;
    plot(get(gca, 'xlim'), median(auc(:, 1, iDoN))*ones(1, 2), '--');  
    ah.YLim =[0.2 1];
    xticklabels({'FICMB', 'IMB(Y)', 'MB(Y)'})
end
%%
% make calibration plot
cmbp = reshape(cmbProbs, [], 1);
mbep = reshape(mbeProbs, [], 1);
mbp = reshape(mbProbs, [], 1);
trueEv = reshape(trueEvents, [], 1);

[nPointsCMB, edges, indsCMB] = histcounts(cmbp, [0:0.2:1]);
[nPointsMBe, ~, indsMBe] = histcounts(mbep, [0:0.2:1]);
[nPointsMB, ~, indsMB] = histcounts(mbp, [0:0.2:1]);

edges = edges(2:end)-0.05;
for iEdge =1:length(edges)
    if nPoints(iEdge)==0
        continue;
    end
    npcmb(iEdge) = sum(trueEv(indsCMB==iEdge))/nPointsCMB(iEdge);
    npmbe(iEdge) = sum(trueEv(indsMBe==iEdge))/nPointsMBe(iEdge);
    npmb(iEdge) = sum(trueEv(indsMB==iEdge))/nPointsMB(iEdge);

end
figure;
scatter(edges, npcmb,'filled');hold on;pause;
scatter(edges, npmbe, 'filled');pause;
scatter(edges, npmb, 'filled');

refline([1,0]);
plot(xlim, [.5 .5], '-k','LineWidth',1,'HandleVisibility','off');
plot([0.5 0.5], ylim, '-k','LineWidth',1,'HandleVisibility','off');



%%
for i=1:3
    bp{i}= squeeze(auc(:, i, :))

end

%         for iS=1:1000
%             zconfs_all = expDsTest.data(iS, [1 3:nVars])+1;
%             zconfs_adj_ems =expDsTest.data(iS, [1 zbest_ems])+1;
% 
%             indxcell_all = num2cell([1 zconfs_all],1);
%             indxcell = indxcell_all([1 zconfs_exp]);
%             indxcell_em = indxcell_all([1 zconfs_adj_ems]);
% 
%             probs(iS, 2) = pydoxzFAS(sub2ind(size(pydoxzFAS),indxcell{:})); 
%             probs(iS, 1) = idEst(1, zconfs_exp(1));
%             probs(iS, 3) = pydoxzAll(sub2ind(size(pydoxzAll),indxcell_all{:}));
%             probs(iS, 4) = pydoxzFASjt(1, ismember(zconfs+1, zconfs_exp, 'rows'));
%             probs(iS, 5) = pydoxzFASemjt(1, ismember(zemconfs+1, zconfs_adj_ems, 'rows')); 
%             probs(iS, 6) = pydoxzalljt(1, ismember(allconfs+1, zconfs_all,'rows'));
% 
%             end
%         end
%     end
% 
% end
%% 
close all;
figure;
[~, b] =  max(scores);
for iDoN=1:nDoNs
    subplot(1, nDoNs, iDoN);boxplot(diffs(1:100), squeeze(b(1, iDoN, :)));hold all
end
%%
