function indwithY = findIndWithYgivCMB(Y, cmb, smm)
% finds Z: Y\ind C|Z in G_{\overline X}.

smm_x = manipulatesmm(smm, 1);
indwithY = findmseparations(smm_x, Y, cmb);
end