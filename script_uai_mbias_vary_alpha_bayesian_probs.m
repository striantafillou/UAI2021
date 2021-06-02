% tetrad_run_mfbs
% test the new algorithm
clear; clc;
comp ='sot16.PITT'; 
%comp ='sofia

code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs'];
addpath(genpath(code_path));
code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_effects'];
addpath(genpath(code_path));
code_path = ['C:\Users\' comp '\Dropbox\MATLAB\UAI2021'];
addpath(genpath(code_path));
% control panel
N=10000;x=1; y=2;
doNs =[500]; maxDoN = doNs(end);nDoNs =length(doNs);
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


nIters=20;
%scores = nan(nSets*2, nDoNs, nIters);

% initialize
auc = nan(3, 10);
diffs = nan(nIters,1);
as=[0.5:0.05:1];
ia=0;
for a=as
    ia=ia+1;
    %Age
    nodesOr{4}.cpt = [.8 .2]';
    % 5: Gender F/M
    nodesOr{5}.cpt = [.9 .1]';
    % 3. noisy and
    nodesOr{3}.cpt(:, 1,1) =[a 1-a];
    nodesOr{3}.cpt(:, 2,1) =[0 1];
    nodesOr{3}.cpt(:, 1,2) =[0 1];
    nodesOr{3}.cpt(:, 2,2) =[0 1];
    % Outcome (no, yes) treatment (no/yes), gender(F/M)
    nodesOr{2}.cpt(:, 1,1) =[a 1-a];
    nodesOr{2}.cpt(:, 1,2) =[0 1];
    nodesOr{2}.cpt(:, 2,1) =[0 1];
    nodesOr{2}.cpt(:, 2,2) =[0 1];
    % treatment no/yes giv age
    nodesOr{1}.cpt(:, 1) =[a,1-a];
    nodesOr{1}.cpt(:, 2) =[0 1];
    isLatent  = [false false false true true];
    
    [tIM] = tetradEIM(dag, nodesOr, domainCountsOr);
    jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tIM);
    idTrue = estimateDoProbCondJT(2,1,3,dag, nodesOr, domainCountsOr);
    idBiased = estimateCondProbJT(2, [1 3], jtalg, nVarsOr, domainCountsOr);
    jtalg_do_true = tetradJtalgDo(dag,nodesOr,domainCountsOr, 1);

   % [~, confProbs] = estimateJointProbJT([1 3], jtalg, nVarsOr, domainCountsOr)
    bias(ia,2)= idBiased(1,3)-idTrue(1,3); 
    obsDatasetOr = simulatedata(nodesOr,N, 'discrete', 'domainCounts', domainCountsOr, 'isLatent', isLatent); % simulate data from original distribution
    % observational data, learn dag, and BN
    obsDataset = subdataset(obsDatasetOr);domainCounts=obsDataset.domainCounts;    
    [adag, dsj]= tetradFges12(obsDataset.data, domainCounts, 'pretreat', true, 'onlyY', true);
    mby = setdiff(find(adag(:, 2)), 1)';
    [sets, nSets] = allSubsets(nVars,mby);
    sets(:, x) = true;
    [logpDegivDoHw, logpDegivDoHn] =deal(-inf*ones(nSets,nDoNs));    
    logpDogivH = nan(nSets,1);
    [curSet, Nexp, Nobs,cmbconfigs, nConfigs,pobs, probsExp] = deal(cell(1, nSets));
    for iSet=1:nSets
        %fprintf('iSet %d-----------\n', iSet)
        curSet{iSet} = find(sets(iSet,:));
        [pobs{iSet}, ~, cmbconfigs{iSet}, Nobs{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], obsDataset.data, domainCounts(1:nVars));
        nConfigs{iSet} = size(cmbconfigs{iSet}, 1);
        logpDogivH(iSet) = dirichlet_bayesian_score(Nobs{iSet});
    end
    %Nobs{end}
        % testdata
    testDsOr = simulateDoData(nodesOr, 1,  0:domainCountsOr(1)-1, 1000, 'discrete', 'domainCounts', domainCounts);    
    testDs = subdataset(testDsOr,isLatent);
    testData = testDs.data;
    trueProbs(:, ia) = estimateTruePs(testData, jtalg_do_true, [1 3]);

    for iter=1:nIters
        % simulate experimental
        expDsOr = simulateDoData(nodesOr, 1,  0:domainCountsOr(1)-1, maxDoN, 'discrete', 'domainCounts', domainCounts);
        expDs = subdataset(expDsOr,isLatent);
        expData = expDs.data;

        % find IMB from De
        [adage]= tetradFges12(expDs.data,domainCounts, 'pretreat', true, 'onlyY', true);
        mbye = find(adage(:, 2))';%setdiff(find(adage(:, 2)), 1)';
        [probsmbye, ~, mbyeconfigs] = cond_prob_mult_inst(y, mbye, expData, domainCounts(1:nVars));

        % fIMB
        for iSet=1:nSets
            curSet{iSet} = find(sets(iSet,:));
            [probsExp{iSet}, ~, ~, Nexp{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], expData, domainCounts(1:nVars));            
            [logpDegivDoHn(iSet, 1), score_n{iSet}] = dirichlet_bayesian_score(Nexp{iSet});
        end
        [logpDegivDoHw(iSet, 1),score_in{iSet}] = dirichlet_bayesian_score(Nexp{iSet}, Nobs{iSet});
        
        numer = [logpDegivDoHw(:, 1)+logpDogivH;logpDegivDoHn(:, 1)+logpDogivH];
        denom =sumOfLogsV(numer);
        scores = numer-denom;
        nConfigs = size(cmbconfigs{nSets}, 1);
        bprobs = zeros(nSets*2, domainCounts(y), nConfigs);
        bprobs_av =zeros(domainCounts(y), nConfigs);
        for iSet=1:nSets
            bprobs(nSets+iSet, :, :)= reshapeProbs(probsExp{iSet}, sets(iSet,:), sets(nSets, :), cmbconfigs{iSet}, cmbconfigs{nSets}, nVars, domainCounts, y);
        end
        bprobs(nSets, :, :)= reshapeProbs(dirichlet_posterior_expectation(Nexp{nSets}, Nobs{nSets}), sets(iSet,:), sets(nSets, :), cmbconfigs{iSet}, cmbconfigs{nSets}, nVars, domainCounts, y);
        for iConfig =1:nConfigs
            for iy =1:domainCounts(y)
            bprobs_av(iy, iConfig) = exp(sumOfLogsV(scores+ log(bprobs(:, iy, iConfig))));
            end
        end
        

        [~, b] =max(scores);
%         bs(iter,ia) =b;
         if b<nSets+1
             bestSet = curSet{b};
%             configs = cmbconfigs{b};
%             bestprobs = dirichlet_posterior_expectation(Nexp{b}, Nobs{b});
         else
             bestSet = curSet{b-nSets};
%             configs = cmbconfigs{b-nSets};
%             bestprobs = probsExp{b-nSets};
         end
        [auc(iter, 1, ia), sens(iter, 1, ia), spec(iter, 1, ia), probs_1(:, ia, iter)]= estimateMetrics(bestSet, bprobs_av, cmbconfigs{end}, testData(1:1000, :));
        [auc(iter, 2, ia), sens(iter, 2, ia), spec(iter, 2, ia), probs_2(:, ia, iter)]= estimateMetrics(mbye, probsmbye, mbyeconfigs, testData(1:1000, :));
        [auc(iter, 3, ia), sens(iter, 3, ia), spec(iter, 3, ia), probs_3(:, ia, iter)] = estimateMetrics(curSet{end},pobs{end},  cmbconfigs{end}, testData(1:1000, :));
    end
end
 
close all;
figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_bias_alpha'] ;%
close all;fig =figure;colors = colormap(lines);hold all;
markers ={'s', '*', 'p'};
ah = gca;
hold all;
plot(as, mean(squeeze(mean(abs(trueProbs(1:1000, :)-probs_1), 1))'), '-', 'LineWidth', 2, 'Color', colors(4, :));
plot(as, mean(squeeze(mean(abs(trueProbs(1:1000, :)-probs_2), 1))'), '-', 'LineWidth', 2, 'Color', colors(3, :));

plot(as, mean(squeeze(mean(abs(trueProbs(1:1000, :)-probs_3), 1))'), '-', 'LineWidth', 2, 'Color', colors(2, :));

xlabel('$\alpha$ (increasing bias)', 'Interpreter', 'latex')
ylabel('absolute bias for $P(Y|do(X=1), M)$', 'Interpreter', 'latex')
    set(ah,'fontsize',14,'FontWeight','bold')
    set([ah.YAxis, ah.XAxis], ...
    'FontName'   , 'Helvetica', ...
    'FontSize'   , 16  );% % xticklabels({'FGES-MB$(D_e)$', 'FGES-MB$(D_o)$', 'FCI-IMB'},'interpreter', 'latex')
    set(gca,'TickLabelInterpreter','latex');
legend({'FindIMB', 'IMB', 'OMB'}, 'Interpreter', 'latex', 'Location', 'NorthWest')

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
%saveas(fig, ['C:/Users/sot16.PITT/Dropbox/Apps/Overleaf/Causal Markov Blankets/figures/' figName], 'pdf')


%%

figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_bias_alpha'] ;%
close all;fig =figure;colors = colormap(lines);hold all;
markers ={'s', '*', 'p'};
ah = gca;
for iAlg =[1 3]
    scatter(-1, -1, 'Marker', markers{iAlg});
    %plot([0.05:0.05:1], mean(squeeze(sens(:, iAlg, :))), '-', 'LineWidth', 2, 'Color', colors(iAlg, :), 'Marker', markers{iAlg});
    plot([0.05:0.05:1], mean(squeeze(sens(:, iAlg, :))), '-', 'LineWidth', 2, 'Color', colors(iAlg, :), 'Marker', markers{iAlg});
    ah.XLim =[0 1]; ah.YLim =[0 1];pause(1)
end
xlabel(['\alpha (increasing bias)'])
ylabel('specificity')
    set(ah,'fontsize',14,'FontWeight','bold')
    set([ah.YAxis, ah.XAxis], ...
    'FontName'   , 'Helvetica', ...
    'FontSize'   , 12  );% % xticklabels({'FGES-MB$(D_e)$', 'FGES-MB$(D_o)$', 'FCI-IMB'},'interpreter', 'latex')

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
saveas(fig, ['C:/Users/sot16.PITT/Dropbox/Apps/Overleaf/Causal Markov Blankets/figures/' figName], 'pdf')
%%
figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_bias_alpha'] ;%
close all;fig =figure;colors = colormap(lines);hold all;
markers ={'s', '*', 'p'};
ah = gca;
for iAlg =[1 3]
    scatter(-1, -1, 'Marker', markers{iAlg});
    %plot([0.05:0.05:1], mean(squeeze(sens(:, iAlg, :))), '-', 'LineWidth', 2, 'Color', colors(iAlg, :), 'Marker', markers{iAlg});
    plot([0.05:0.05:1], median(squeeze(spec(:, iAlg, :))), '-', 'LineWidth', 2, 'Color', colors(iAlg, :), 'Marker', markers{iAlg});
    ah.XLim =[0 1]; ah.YLim =[0 1];
end
xlabel(['\alpha (increasing bias)'])
ylabel('specificity')
    set(ah,'fontsize',14,'FontWeight','bold')
    set([ah.YAxis, ah.XAxis], ...
    'FontName'   , 'Helvetica', ...
    'FontSize'   , 12  );% % xticklabels({'FGES-MB$(D_e)$', 'FGES-MB$(D_o)$', 'FCI-IMB'},'interpreter', 'latex')

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
saveas(fig, ['C:/Users/sot16.PITT/Dropbox/Apps/Overleaf/Causal Markov Blankets/figures/' figName], 'pdf')




% % 
% %     colors = colormap(lines);
% % 
% %     h = findobj(ah,'Tag','Box');lh = length(h);
% %     patch(get(h(1),'XData'),get(h(1),'YData'),colors(1,:),'FaceAlpha',.5);
% %     patch(get(h(2),'XData'),get(h(2),'YData'),colors(2,:),'FaceAlpha',.5);
% %     patch(get(h(3),'XData'),get(h(3),'YData'),colors(3,:),'FaceAlpha',.5);
% %     patch(get(h(4),'XData'),get(h(4),'YData'),colors(4,:),'FaceAlpha',.5);
% %     
% %     
% %     hAxes = gca;
% %     a = get(hAxes,'XTickLabel');  
% %     set(hAxes,'XTickLabel',a,'fontsize',14,'FontWeight','bold')
% %     set(hAxes.Title,'fontsize',18,'FontWeight','bold')
% %     set([hAxes.YAxis, hAxes.XAxis], ...
% %     'FontName'   , 'Helvetica', ...
% %     'FontSize'   , 12          );
% % end
% % 
% % fig = gcf;
% % 
% % fig.PaperPositionMode = 'auto';
% % fig_pos = fig.PaperPosition;
% % fig.PaperSize = [fig_pos(3) fig_pos(4)];
% % 
% % 
% % %
% %     
% % 
% % 
% % 
% % for iter=1:nIters
% %     fprintf('Iter %d \n', iter);
% %     initialize
% %   
% % 
% %    mby = findMBsmm(smm,y);mby = setdiff(mby, 1);
% % 
% %    diffs(iter) =mean(abs(idTruePop(1, :)-idBiased(1, :)));
% %     simulate data with dummy nodes.
% %    
% %     
% %     dTruePop = estimateDoProbCondJT(2,1,mby,dag, nodesOr, domainCountsOr);
% %     idBiased = estimateCondProbJT(2, [1 mby], jtalg, nVarsOr, domainCountsOr);
% % 
% %     
% %     [sets, nSets] = allSubsets(nVars,mby);
% %     sets(:, x) = true;
% %     [logpDegivDoHw, logpDegivDoHn] =deal(nan(nSets,nDoNs));    
% %     logpDogivH = nan(nSets,1);
% %     [curSet, Nexp, Nobs,cmbconfigs, nConfigs,pobs, probsExp] = deal(cell(1, nSets));
% %     for iSet=1:nSets
% %         fprintf('iSet %d-----------\n', iSet)
% %         curSet{iSet} = find(sets(iSet,:));
% %         [pobs{iSet}, ~, cmbconfigs{iSet}, Nobs{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], obsDataset.data, domainCounts(1:nVars));
% %         nConfigs{iSet} = size(cmbconfigs{iSet}, 1);
% %         logpDogivH(iSet) = dirichlet_bayesian_score(Nobs{iSet});
% %     end
% %     simulate experimental data 
% % 
% %         
% %         for iSet=1:nSets
% %             curSet{iSet} = find(sets(iSet,:));
% %             [probsExp{iSet}, ~, ~, Nexp{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], expData(doInds(:, 1), :), domainCounts(1:nVars));            
% %             logpDegivDoHw(iSet, 1) = dirichlet_bayesian_score(Nexp{iSet}, Nobs{iSet});
% %             logpDegivDoHn(iSet, 1) = dirichlet_bayesian_score(Nexp{iSet});
% %         end
% %         numer = [logpDegivDoHw(:, 1)+logpDogivH;logpDegivDoHn(:, 1)+logpDogivH];
% %         denom =sumOfLogsV(numer);
% %         scores = numer-denom;
% %         [~, b] =max(scores);
% %         if b<nSets+1
% %             bestSet = curSet{b};
% %             configs = cmbconfigs{b};
% %             bestprobs = dirichlet_posterior_expectation(Nexp{b}, Nobs{b});
% %         else
% %             bestSet = curSet{b-nSets};
% %             configs = cmbconfigs{b-nSets};
% %             bestprobs = probsExp{b-nSets};
% %         end
% %         auc(iter,1, ia)= estimateAUCfs(bestSet, bestprobs, configs, testData);
% %         auc(iter,2, ia) = estimateAUCfs(mbye, probsmbye, mbyeconfigs, testData);
% %         auc(iter,3, ia) = estimateAUCfs(curSet{end},pobs{end},  cmbconfigs{end}, testData);
% %      
% % 
% %         against dexp all
% %     end
% % end
% % 
% % 
% % close all;
% % figure;hold all;
% % for iDoN=1:nDoNs
% %     subplot(1, nDoNs,iDoN); ah = gca;boxplot(auc(:,:, iDoN)); hold all;
% %     plot(get(gca, 'xlim'), median(auc(:, 1, iDoN))*ones(1, 2), '--');  
% %     ah.YLim =[0.2 1];
% %     xticklabels({'FICMB', 'IMB(Y)', 'MB(Y)'})
% % end
% % 
% % make calibration plot
% % cmbp = reshape(cmbProbs, [], 1);
% % mbep = reshape(mbeProbs, [], 1);
% % mbp = reshape(mbProbs, [], 1);
% % trueEv = reshape(trueEvents, [], 1);
% % 
% % [nPointsCMB, edges, indsCMB] = histcounts(cmbp, [0:0.2:1]);
% % [nPointsMBe, ~, indsMBe] = histcounts(mbep, [0:0.2:1]);
% % [nPointsMB, ~, indsMB] = histcounts(mbp, [0:0.2:1]);
% % 
% % edges = edges(2:end)-0.05;
% % for iEdge =1:length(edges)
% %     if nPoints(iEdge)==0
% %         continue;
% %     end
% %     npcmb(iEdge) = sum(trueEv(indsCMB==iEdge))/nPointsCMB(iEdge);
% %     npmbe(iEdge) = sum(trueEv(indsMBe==iEdge))/nPointsMBe(iEdge);
% %     npmb(iEdge) = sum(trueEv(indsMB==iEdge))/nPointsMB(iEdge);
% % 
% % end
% % figure;
% % scatter(edges, npcmb,'filled');hold on;pause;
% % scatter(edges, npmbe, 'filled');pause;
% % scatter(edges, npmb, 'filled');
% % 
% % refline([1,0]);
% % plot(xlim, [.5 .5], '-k','LineWidth',1,'HandleVisibility','off');
% % plot([0.5 0.5], ylim, '-k','LineWidth',1,'HandleVisibility','off');
% % 
% % 
% % 
% % 
% % for i=1:3
% %     bp{i}= squeeze(auc(:, i, :))
% % 
% % end
% % 
% %         for iS=1:1000
% %             zconfs_all = expDsTest.data(iS, [1 3:nVars])+1;
% %             zconfs_adj_ems =expDsTest.data(iS, [1 zbest_ems])+1;
% % 
% %             indxcell_all = num2cell([1 zconfs_all],1);
% %             indxcell = indxcell_all([1 zconfs_exp]);
% %             indxcell_em = indxcell_all([1 zconfs_adj_ems]);
% % 
% %             probs(iS, 2) = pydoxzFAS(sub2ind(size(pydoxzFAS),indxcell{:})); 
% %             probs(iS, 1) = idEst(1, zconfs_exp(1));
% %             probs(iS, 3) = pydoxzAll(sub2ind(size(pydoxzAll),indxcell_all{:}));
% %             probs(iS, 4) = pydoxzFASjt(1, ismember(zconfs+1, zconfs_exp, 'rows'));
% %             probs(iS, 5) = pydoxzFASemjt(1, ismember(zemconfs+1, zconfs_adj_ems, 'rows')); 
% %             probs(iS, 6) = pydoxzalljt(1, ismember(allconfs+1, zconfs_all,'rows'));
% % 
% %             end
% %         end
% %     end
% % 
% % end
% % 
% % close all;
% % figure;
% % [~, b] =  max(scores);
% % for iDoN=1:nDoNs
% %     subplot(1, nDoNs, iDoN);boxplot(diffs(1:100), squeeze(b(1, iDoN, :)));hold all
% % end
% % 
