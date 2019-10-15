% Compiles all mex routines that are part of toolbox.
%
% USAGE
%  toolboxCompile
%
% INPUTS
%
% OUTPUTS
%
% EXAMPLE
%
% See also
%
% Piotr's Image&Video Toolbox      Version 3.22
% Copyright 2013 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see external/bsd.txt]

% compile options including openmp support for C++ files
opts = {'-output'};
if(exist('OCTAVE_VERSION','builtin')), opts={'-o'}; end
if( ispc ), optsOmp={'OPTIMFLAGS="$OPTIMFLAGS','/openmp"'}; else
  optsOmp={'CXXFLAGS="\$CXXFLAGS','-fopenmp"'};
  optsOmp=[optsOmp,'LDFLAGS="\$LDFLAGS','-fopenmp"'];
end

% list of files (missing /private/ part of directory)
fs={'classify/binaryTreeTrain1.cpp', 'classify/forestInds.cpp'};
n=length(fs); useOmp=zeros(1,n); useOmp([1 2])=1;

% compile every funciton in turn (special case for dijkstra)
disp('Compiling Piotr''s Toolbox.......................');
rd=fileparts(mfilename('fullpath')); rd=rd(1:end-9); tic;
try
  for i=1:n
    [d,f1,e]=fileparts(fs{i}); f=[rd '/' d '/private_tmp/' f1];
    if(useOmp(i)), optsi=[optsOmp opts]; else optsi=opts; end
    fprintf(' -> %s\n',[f e]); mex([f e],optsi{:},[f '.' mexext]);
  end
catch ME
	disp(ME);
  fprintf(['C++ mex failed, likely due to lack of a C++ compiler.\n' ...
    'Run ''mex -setup'' to specify a C++ compiler if available.\n'...
    'Or, one can specify a specific C++ explicitly (see mex help).\n']);
end
disp('..................................Done Compiling'); toc;
