% MTEX reference workflow (to execute in MATLAB with MTEX configured).
% Run on exported verification_roi.ctf and compare matched ROI maps, thresholds,
% orientation reference frames and phase dictionaries before publication.
% This script is an interface, NOT evidence of MTEX validation.

% Set appropriate crystal specimen and convention for your data here.
CS = { 'notIndexed', ...
  crystalSymmetry('6/mmm',[2.951 2.951 4.684],'X||a*','Y||b','Z||c','mineral','Titanium-Hexagonal'), ...
  crystalSymmetry('m-3m',[3.311 3.311 3.311],'mineral','Titanium-Cubic')};
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','outOfPlane');
file = fullfile(fileparts(mfilename('fullpath')),'verification_roi.ctf');
assert(isfile(file),'Place verification_roi.ctf beside this script');
ebsd=EBSD.load(file,CS,'interface','ctf','convertEuler2SpatialReferenceFrame');
[grains,ebsd.grainId]=calcGrains(ebsd('indexed'),'angle',5*degree);
grains=smooth(grains,3);
figure;plot(ebsd('indexed'),ebsd('indexed').orientations);hold on;plot(grains.boundary,'lineWidth',0.6);
figure;plot(ebsd('indexed'),ebsd('indexed').KAM./degree);
% For more accurate prior-beta grain reconstruction use:
% https://mtex-toolbox.github.io/ParentChildVariants_py.html
% Requires grain graph fitting and Burgers OR calibration; not automatic.
