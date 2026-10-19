% Prior-beta reconstruction (MTEX reference implementation, requires MATLAB+MTEX).
% Source: MTEX official TiBetaReconstruction example. This script is NOT
% executed or verified in the current delivery environment.
% Verify Euler/sample reference transforms and exact phase mineral names!

mtexfile = fullfile(fileparts(mfilename('fullpath')),'verification_roi.ctf');
assert(isfile(mtexfile),'Place verification_roi.ctf next to this script');
CS = {'notIndexed', ...
    crystalSymmetry('6/mmm',[2.951 2.951 4.684],'X||a*','Y||b','Z||c', 'mineral','Titanium-Hexagonal'), ...
    crystalSymmetry('m-3m',[3.311 3.311 3.311], 'mineral','Titanium-Cubic')};
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','outOfPlane');
ebsd = EBSD.load(mtexfile, CS, 'interface','ctf','convertEuler2SpatialReferenceFrame');
% Low angle threshold for parent graph reconstruction: see MTEX official TiBetaReconstruction page
[grains,ebsd] = calcGrains(ebsd,'threshold',1.5*degree,'removeQuadruplePoints');
job = parentGrainReconstructor(ebsd,grains);
job.p2c = orientation.Burgers(job.csParent,job.csChild);
job.calcVariantGraph('threshold',1.5*degree);
job.clusterVariantGraph('numIter',3);
job.calcParentFromVote;
figure;plot(job.parentGrains,'ipfDirection',zvector);
title('MTEX prior beta reconstruction - requires manual QC');
saveas(gcf, fullfile(fileparts(mfilename('fullpath')),'MTEX_prior_beta_candidate.png'));
% For per-pixel variant / parent ids, consult job.calcParentEBSD
% fit / confidence checks must be performed before publishing.
