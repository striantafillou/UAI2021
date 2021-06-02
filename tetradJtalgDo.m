function jtalg = tetradJtalgDo(dag, nodes,domainCounts, x)
    doNodes = nodes; 
    doNodes{x}.parents=[];doNodes{x}.cpt = ones(domainCounts(x), 1);
    doDag = dag;doDag(:, x)=0;doDag = makeConnected(doDag);
    [tIMdo] = tetradEIM(doDag, doNodes, domainCounts);
    jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tIMdo); % selection posterior
end