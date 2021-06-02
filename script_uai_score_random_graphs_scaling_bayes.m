% test the new algorithm
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
doNs =[50 100 500]; maxDoN = doNs(end);nDoNs =length(doNs);
doInds = getExperimentalInds(maxDoN, doNs, nDoNs, 2);
% simulate data from X->Y, X<-W1->W2->Y
% nVarsOr =5;
% dag = zeros(nVarsOr);
% dag(1, 2)=1; dag(3, 1) =1; dag(3, 4)=1; dag(4, 2)=1;dag(5, 2)=1;
% isLatent = [false false false false false];
%load twoconfs.mat
%load mbiasplusgraph.mat  
nVarsOr =60;
isLatent = false(1, nVarsOr); 
isLatent(41:60)=true;
%load confandemgraph.mat
nVars= sum(~isLatent);
domainCountsOr =[2 2 randi([2 2], 1, nVarsOr-2)];


nIters=50;
% initialize
[auc, sens, spec] = deal(nan(nIters,3, nDoNs));
[cmbProbs, mbeProbs, mbProbs,imboProbs] = deal(nan(200, nIters, nDoNs));
[trueProbs, trueEvents] = deal(nan(200, nIters));
diffs = nan(nIters,1);
timesObs = nan(nIters, 1);
timesExp = nan(nIters,1);
timesFCI = nan(nIters,1);
saveFolder = 'C:/Users/sot16.PITT/Dropbox/MATLAB/causal_effects/results/uai2021';
if(~isdir(saveFolder));mkdir(saveFolder);end

imbys_true =false(nIters, nVars);
[imbys_fci, imbys_fimb, imbys_e] = deal(false(nIters, nVars, nDoNs));

fName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_res'] ;%
%%
for iter=1:50
    totalt =tic;
    fprintf('Iter %d \t', iter);
    
    dag = randomDagWith12PreTreatmentConf(nVarsOr, 3);    
    % nodesOr: with confounders.
    [nodesOr, domainCountsOr, orderOr, rnodes] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCountsOr, 'minNumStates', domainCountsOr);
    dagx = dag;dagx(:, x)=0;smmx = dag2smm(dagx, isLatent);
    imby_true =findMBsmm(smmx,y);imby = setdiff(imby_true, 1);
    imbys_true(iter, imby)=true;
    
    %jtalg_do_true = tetradJtalgDo(dag,nodesOr,domainCountsOr, 1);
    % simulate data with dummy nodes.
    obsDatasetOr = simulatedata(nodesOr,N, 'discrete', 'domainCounts', domainCountsOr, 'isLatent', isLatent); % simulate data from original distribution
 
    
    % observational data, learn dag, and BN
    obsDataset = subdataset(obsDatasetOr);domainCounts=obsDataset.domainCounts;   

    % simulate experimental data 
    expDsOr = simulateDoData(nodesOr, 1,  0:domainCountsOr(1)-1, maxDoN, 'discrete', 'domainCounts', domainCounts);
    expDs = subdataset(expDsOr,isLatent);
    expData = expDs.data;
    
    % simulate test data
    testDsOr = simulateDoData(nodesOr, 1,  0:domainCountsOr(1)-1, 100, 'discrete', 'domainCounts', domainCounts);    
    testDs = subdataset(testDsOr,isLatent);
    testData = testDs.data;
    
 %   trueProbs(:, iter) = estimateTruePs(testData, jtalg_do_true, imby_true);
    
    to=tic;
    [adag, dsj]= tetradFges12(obsDataset.data, domainCounts, 'pretreat', true, 'onlyY', true);
    mby = setdiff(find(adag(:, 2)), 1)';
    timesObs(iter) = toc(to);
    fprintf(' | Time elapsed %.3f\t',toc(totalt));
    te=tic;
    [sets, nSets] = allSubsets(nVars,mby);
    sets(:, x) = true;
    [logpDegivDoHw, logpDegivDoHn] =deal(nan(nSets,nDoNs));    
    logpDogivH = nan(nSets,1);
    [curSet, Nexp, Nobs,cmbconfigs, nConfigs, pobs, probsExp] = deal(cell(1, nSets));
    for iSet=1:nSets
        %fprintf('iSet %d-----------\n', iSet)
        curSet{iSet} = find(sets(iSet,:));
        [pobs{iSet}, ~, cmbconfigs{iSet}, Nobs{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], obsDataset.data, domainCounts(1:nVars));
        nConfigs{iSet} = size(cmbconfigs{iSet}, 1);
    end

    logpDogivH(:) = dirichlet_bayesian_score(Nobs{iSet});
    timesExp(iter, 1) =toc(te);
    for iDoN =1:nDoNs
        ti =tic;
        [adage]= tetradFges12(expData(doInds(:, iDoN), :), domainCounts, 'pretreat', true, 'onlyY', true);
        mbye = find(adage(:, 2))';%setdiff(find(adage(:, 2)), 1)';
        [probsmbye, ~, mbyeconfigs] = cond_prob_mult_inst(y, mbye, expData(doInds(:, iDoN), :), domainCounts(1:nVars));
         for iSet=1:nSets
            %curSet{iSet} = find(sets(iSet,:));
            logpDegivDoHw(iSet, iDoN) =-inf;
            [probsExp{iSet}, ~, ~, Nexp{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], expData(doInds(:, iDoN), :), domainCounts(1:nVars));            
            logpDegivDoHn(iSet, iDoN) = dirichlet_bayesian_score(Nexp{iSet});
        end
        logpDegivDoHw(iSet, iDoN) = dirichlet_bayesian_score(Nexp{iSet}, Nobs{iSet});

        numer = [logpDegivDoHw(:, iDoN)+logpDogivH;logpDegivDoHn(:, iDoN)+logpDogivH];
        denom =sumOfLogsV(numer);
        scores = numer-denom;
        %s(1:nSets-1) = -inf;
        
        [bprobs_av, bprobs] = estimateAllProbsTab(sets, nSets, cmbconfigs, Nexp, Nobs, probsExp, domainCounts, y, nVars, scores);
        [~, b] =max(scores);
        if b<nSets+1
            bestSet = curSet{b};
           % configs = cmbconfigs{b};
            %bestprobs = dirichlet_posterior_expectation(Nexp{b}, Nobs{b});
        else
            bestSet = curSet{b-nSets};
            %configs = cmbconfigs{b-nSets};
            %bestprobs = probsExp{b-nSets};
        end
        timesExp(iter, 1+iDoN) =toc(ti);
        %oracle
%         tic
%         [imbyfci, sameasCMB, pag]= contextFCI(obsDataset, expData);
%         imbys_fci(iter, imbyfci)=true;
%         if length(imbyfci)<11            
%             if sameasCMB
%                 [impbopobs, ~, imboconfigs] = cond_prob_mult_inst(y, [x imbyfci],[obsDataset.data; expData(doInds(:, iDoN), :)], domainCounts(1:nVars));
%             else
%                 [impbopobs, ~, imboconfigs] = cond_prob_mult_inst(y, [x imbyfci],[expData(doInds(:, iDoN), :)], domainCounts(1:nVars));
%             end
%             [auc(iter,4, iDoN), sens(iter, 4, iDoN),  spec(iter, 4, iDoN), imboProbs(:, iter, iDoN)] = estimateMetrics([x imbyfci], impbopobs,  imboconfigs, testData);
%         end
%         timesFCI(iter)=toc;
        %fprintf(' | Time elapsed %.3f\t',toc(totalt));
        imbys_e(iter, mbye, iDoN)=true;
       % imbys_fci(iter, imbyfci,iDoN) = true;
        imbys_fimb(iter, bestSet, iDoN)=true;
        
        [auc(iter,1, iDoN), sens(iter, 1, iDoN),  spec(iter, 1, iDoN), cmbProbs(:, iter, iDoN)]= estimateMetrics(curSet{end}, bprobs_av, cmbconfigs{end}, testData);
        [auc(iter,2, iDoN), sens(iter, 2, iDoN),  spec(iter, 2, iDoN), mbeProbs(:, iter, iDoN)] = estimateMetrics(mbye, probsmbye, mbyeconfigs, testData);
        [auc(iter,3, iDoN), sens(iter, 3, iDoN),  spec(iter, 3, iDoN), mbProbs(:, iter, iDoN)] = estimateMetrics(curSet{end},pobs{end},  cmbconfigs{end}, testData);

        trueEvents(:,iter) = 1-testData(:, 2);

        % against dexp all
        %save([saveFolder filesep 'nVars_' num2str(nVars) '_N' ...
        %   sprintf('%d',floor(N./1000)) 'K_doN_' num2str(doNs(iDoN)) '_iter_' num2str(iter) '.mat']);
    end

    fprintf(' | Time elapsed %.3f, nSets %d, nSetsOr %d\n',toc(totalt), nSets, length(imby));
   % fprintf('Time elapsed %.3f\n',toc(totalt));
end

%load([saveFolder filesep fName '.mat'])


%%
saveFolder = 'C:\Users\sot16.PITT\Dropbox\Apps\Overleaf\Causal Markov Blankets\figures';
close all;
for iDoN=1:nDoNs
    figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_doN' num2str(doNs(iDoN)) '_auc'] ;%

    fig=figure;hold all;
    ah = gca;boxplot(auc(:,:, iDoN)); 

    plot(get(gca, 'xlim'), median(auc(:,1, iDoN))*ones(1, 2), 'r--');  
    title(['$N_e$: ', num2str(2*doNs(iDoN))],'interpreter', 'latex')  
    xticklabels({'FindIMB','IMB', 'OMB'})
    set(gca,'TickLabelInterpreter','latex');
    ylabel('AUC', 'Interpreter', 'latex')
    colors = colormap(lines);

    h = findobj(ah,'Tag','Box');lh = length(h);
    patch(get(h(1),'XData'),get(h(1),'YData'),colors(2,:),'FaceAlpha',.5);
    patch(get(h(2),'XData'),get(h(2),'YData'),colors(3,:),'FaceAlpha',.5);
    patch(get(h(3),'XData'),get(h(3),'YData'),colors(4,:),'FaceAlpha',.5);
    %patch(get(h(4),'XData'),get(h(4),'YData'),colors(4,:),'FaceAlpha',.5);
    
    [pvals(1) ]=signrank(auc(:,1, iDoN), auc(:,2, iDoN));
    [pvals(2) ]=signrank(auc(:,1, iDoN), auc(:,3, iDoN));
    %[pvals(3) ]=signrank(auc(:,3, iDoN), biasFCIc);
    sigstar({[1 2], [1 3]}, pvals);

    hAxes = gca;
    a = get(hAxes,'XTickLabel');  
    set(hAxes,'XTickLabel',a,'fontsize',14,'FontWeight','bold')
    set(hAxes.Title,'fontsize',18,'FontWeight','bold')
    set([hAxes.YAxis, hAxes.XAxis], ...
    'FontName'   , 'Helvetica', ...
    'FontSize'   , 16         );

    fig = gcf;

    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];

  %  saveas(fig, [saveFolder filesep figName], 'pdf')
end
% fh=gcf;

%%


saveFolder = 'C:\Users\sot16.PITT\Dropbox\Apps\Overleaf\Causal Markov Blankets\figures';
close all;nCols=nDoNs; nRows=1;
for iDoN=1:nDoNs
    figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_doN' num2str(doNs(iDoN)) '_sens'] ;%

    fig=figure;hold all;

    ah = gca;boxplot(sens(:,:, iDoN)); 
    plot(get(gca, 'xlim'), median(sens(:,1, iDoN))*ones(1, 2), 'r--');  
    title(['$N_e$: ', num2str(2*doNs(iDoN))],'interpreter', 'latex')  
    xticklabels({'FIMB','FGES-IMB', 'FGES-MB'})
    set(gca,'TickLabelInterpreter','latex');
    ylabel('Sensitivity', 'Interpreter', 'latex')
    colors = colormap(lines);

    h = findobj(ah,'Tag','Box');lh = length(h);
    patch(get(h(1),'XData'),get(h(1),'YData'),colors(2,:),'FaceAlpha',.5);
    patch(get(h(2),'XData'),get(h(2),'YData'),colors(3,:),'FaceAlpha',.5);
    patch(get(h(3),'XData'),get(h(3),'YData'),colors(4,:),'FaceAlpha',.5);
    %patch(get(h(4),'XData'),get(h(4),'YData'),colors(4,:),'FaceAlpha',.5);
    
    
    hAxes = gca;
    a = get(hAxes,'XTickLabel');  
    set(hAxes,'XTickLabel',a,'fontsize',14,'FontWeight','bold')
    set(hAxes.Title,'fontsize',18,'FontWeight','bold')
    set([hAxes.YAxis, hAxes.XAxis], ...
    'FontName'   , 'Helvetica', ...
    'FontSize'   , 16         );

    fig = gcf;

    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];

    saveas(fig, [saveFolder filesep figName], 'pdf')
end
%%

saveFolder = 'C:\Users\sot16.PITT\Dropbox\Apps\Overleaf\Causal Markov Blankets\figures';
close all;nCols=nDoNs; nRows=1;
for iDoN=1:nDoNs
    figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_doN' num2str(doNs(iDoN)) '_spec'] ;%

    fig=figure;hold all;

    ah = gca;boxplot(spec(:,:, iDoN)); 
    plot(get(gca, 'xlim'), median(spec(:,1, iDoN))*ones(1, 2), 'r--');  
    title(['$N_e$: ', num2str(2*doNs(iDoN))],'interpreter', 'latex')  
    xticklabels({'FindIMB','IMB', 'FGES-MB'})
    set(gca,'TickLabelInterpreter','latex');
    ylabel('Specificity', 'Interpreter', 'latex')
    colors = colormap(lines);
    

    h = findobj(ah,'Tag','Box');lh = length(h);
    patch(get(h(1),'XData'),get(h(1),'YData'),colors(2,:),'FaceAlpha',.5);
    patch(get(h(2),'XData'),get(h(2),'YData'),colors(3,:),'FaceAlpha',.5);
    patch(get(h(3),'XData'),get(h(3),'YData'),colors(4,:),'FaceAlpha',.5);
    %patch(get(h(4),'XData'),get(h(4),'YData'),colors(4,:),'FaceAlpha',.5);
    
    
    hAxes = gca;
    a = get(hAxes,'XTickLabel');  
    set(hAxes,'XTickLabel',a,'fontsize',14,'FontWeight','bold')
    set(hAxes.Title,'fontsize',18,'FontWeight','bold')
    set([hAxes.YAxis, hAxes.XAxis], ...
    'FontName'   , 'Helvetica', ...
    'FontSize'   , 16         );

    fig = gcf;

    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];

    saveas(fig, [saveFolder filesep figName], 'pdf')
end
%%
imbys_e(:, 1:2, :)=false;
imbys_fci(:, 1:2,:) = false;
imbys_fimb(:, 1:2, :)=false;

[sns, spf, fps, tps, fns, tns] = deal(nan(nIters, 3, nDoNs));
for iDoN=1:nDoNs
  for iter=1:nIters
    [sns(iter, 1, iDoN), spf(iter, 1, iDoN), tps(iter, 1, iDoN),  fps(iter, 1, iDoN), tns(iter, 1, iDoN), fns(iter, 1, iDoN)] = confMat(imbys_true(iter, :),imbys_fimb(iter, :, iDoN));
    [sns(iter,2, iDoN), spf(iter, 2, iDoN), tps(iter, 2, iDoN),  fps(iter, 2, iDoN), tns(iter, 2, iDoN), fns(iter,2, iDoN)] = confMat(imbys_true(iter, :),imbys_e(iter, :, iDoN));
%    [sns(iter, 3, iDoN), spf(iter, 3, iDoN), tps(iter, 3, iDoN),  fps(iter, 3, iDoN), tns(iter, 1, iDoN), fns(iter, 3, iDoN)] = confMat(imbys_true(iter, :),mby(iter, :, iDoN));
  end
end

prec = tps./(tps+fps);
rec =  tps./(tps+fns);
% rec(isnan(prec))=1;
% rec(isnan(rec))=1;
colors ={'r', 'g', 'b'};
figure; hold all;
for iAlg=1:3
    color = colors{iAlg};
    for iDoN=1:3
        scatter(nanmean(prec(:, iAlg, iDoN)), nanmean(rec(:, iAlg, iDoN)), color);
    end
end

%%
cmbp = reshape(cmbProbs(:, :, 1), [], 1);
mbep = reshape(mbeProbs(:, :, 1), [], 1);
mbp = reshape(mbProbs(:, :, 1), [], 1);
trueEv = reshape(trueEvents, [], 1);

[nPointsCMB, edges, indsCMB] = histcounts(cmbp, [0:0.1:1]);
[nPointsMBe, ~, indsMBe] = histcounts(mbep, [0:0.1:1]);
[nPointsMB, ~, indsMB] = histcounts(mbp, [0:0.1:1]);

edges = edges(2:end)-0.05;
for iEdge =1:length(edges)

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
close all
figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_spec'] ;%
markers ={'*','s', 'p'};
figure; hold all;

for iAlg=1:3
    plot(mean(squeeze(sens(:, iAlg, :))), mean(squeeze(spec(:, iAlg, :))), 'color', colors(iAlg, :), 'Marker', markers{iDoN}, 'MarkerFaceColor', colors(iAlg, :), 'LineWidth', 2);
end

%%
ah.YLim =[0.2 1];ah=gca;
figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_sens'] ;%
markers ={'o', '*','s', 'p'};
figure; hold all;
for iAlg=1:4
    plot(1:nDoNs, median(squeeze(sens(:, iAlg, :))), 'color', colors(iAlg, :), 'Marker', markers{iDoN}, 'MarkerFaceColor', colors(iAlg, :), 'LineWidth', 2);
end
ah.YLim =[0.2 1]; 


figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_auc'] ;%
figure; hold all;ah=gca;
for iAlg=1:4
    plot(1:nDoNs, median(squeeze(auc(:, iAlg, :))), 'color', colors(iAlg, :), 'Marker', markers{1}, 'MarkerFaceColor', colors(iAlg, :), 'LineWidth', 2);
end
ah.YLim =[0.2 1]; 

%%
  
markers ={'o', '*','s', 'p'};
figure; hold all;
for iAlg=1:4
    plot(1:nDoNs, mean(squeeze(sens(:, iAlg, :))), 'color', colors(iAlg, :), 'Marker', markers{iAlg}');
end

plot(1:nDoNs, median(sens(:, iAlg, iDoN))*ones(1, 2), 'color', colors(iAlg, :), 'Marker', markers{1}');
    plot(1:nDoNs, median(sens(:, iAlg, iDoN))*ones(1, 2), 'color', colors(iAlg, :), 'Marker', markers{1}');  

    title(['$N_e$: ', num2str(2*doNs(iDoN))],'interpreter', 'latex')  
  
    xticklabels({'FIMB', 'FGES-IMB', 'FGES-MB', 'FCI-IMB'})
    set(gca,'TickLabelInterpreter','latex');

    colors = colormap(lines);

    h = findobj(ah,'Tag','Box');lh = length(h);
    patch(get(h(1),'XData'),get(h(1),'YData'),colors(1,:),'FaceAlpha',.5);
    patch(get(h(2),'XData'),get(h(2),'YData'),colors(2,:),'FaceAlpha',.5);
    patch(get(h(3),'XData'),get(h(3),'YData'),colors(3,:),'FaceAlpha',.5);
    patch(get(h(4),'XData'),get(h(4),'YData'),colors(4,:),'FaceAlpha',.5);
    
    
    hAxes = gca;
    a = get(hAxes,'XTickLabel');  
    set(hAxes,'XTickLabel',a,'fontsize',14,'FontWeight','bold')
    set(hAxes.Title,'fontsize',18,'FontWeight','bold')
    set([hAxes.YAxis, hAxes.XAxis], ...
    'FontName'   , 'Helvetica', ...
    'FontSize'   , 12          );


fig = gcf;

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

saveas(gcf, [saveFolder filesep figName], 'pdf')
% fh=gcf;