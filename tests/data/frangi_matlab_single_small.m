function frangi_matlab_single_small(frangi_dir, out_file)
% Writes tests/data/frangi_matlab_single_small.bin: Kroon's FrangiFilter3D (frangi_filter_version2a,
% as QSMART bundles it) on a single-precision 16x18x20 oblique tube, for the unit test
% `matches_matlab_single_precision` in src/utils/frangi.rs.
%
% Recovered 2026-10-30 from the session that made the fixture (Bunya, qsmart-parity/frangi32),
% unchanged except for the paths and the variant below being made here rather than kept as a
% file. Run in licensed MATLAB R2023b with the precompiled eig3volume/imgaussian mex:
%   frangi_matlab_single_small('<frangi_filter_version2a>', 'frangi_matlab_single_small.bin')
% The QSM-CI qsmart-matlab container runs FrangiFilter3D only inside the whole QSMART pipeline
% (QSMxT/QSM-CI#296), so this fixture is not made with it.
%
% FrangiFilter3D_crexp is FrangiFilter3D with its three single-precision exp calls evaluated as
% single(exp(double(.))) (a correctly rounded single exp) instead of MATLAB's single exp.
%
% Layout (little-endian, column-major): the u16 input, then f32 volumes: scales 0.01:0.01:0.05
% with C = 500, as MATLAB returns them and with the correctly rounded exp; scales 0.5:0.5:2 with
% C = 200 with that exp.
addpath(frangi_dir);
src = fileread(fullfile(frangi_dir, 'FrangiFilter3D.m'));
src = strrep(src, 'function [Iout,whatScale,Voutx,Vouty,Voutz]=FrangiFilter3D(I,options)', ...
                  'function [Iout,whatScale,Voutx,Vouty,Voutz]=FrangiFilter3D_crexp(I,options)');
src = strrep(src, '(1-exp(-(Ra.^2./A)))', '(1-single(exp(double(-(Ra.^2./A)))))');
src = strrep(src, 'exp(-(Rb.^2./B))', 'single(exp(double(-(Rb.^2./B))))');
src = strrep(src, '(1-exp(-S.^2./(2*options.FrangiC^2)))', '(1-single(exp(double(-S.^2./(2*options.FrangiC^2)))))');
d = tempname; mkdir(d); fid = fopen(fullfile(d, 'FrangiFilter3D_crexp.m'), 'w'); fwrite(fid, src); fclose(fid);
addpath(d);
% 16x18x20 oblique tube, integer intensities
[X,Y,Z]=ndgrid(0:15,0:17,0:19);
dd=[1 2 3]/norm([1 2 3]); p=cat(4,X-7.5,Y-8.5,Z-9.5);
proj=p(:,:,:,1)*dd(1)+p(:,:,:,2)*dd(2)+p(:,:,:,3)*dd(3);
r2=sum(p.^2,4)-proj.^2;
rng(7); I=uint16(round(2500*exp(-r2/(2*1.2^2)))+randi([0 60],size(r2)));
Is=single(I);
o1=struct('FrangiScaleRange',[0.01 0.05],'FrangiScaleRatio',0.01,'FrangiAlpha',0.5,'FrangiBeta',0.5,'FrangiC',500,'verbose',false,'BlackWhite',false);
o2=struct('FrangiScaleRange',[0.5 2],'FrangiScaleRatio',0.5,'FrangiAlpha',0.5,'FrangiBeta',0.5,'FrangiC',200,'verbose',false,'BlackWhite',false);
V1=FrangiFilter3D(Is,o1); V1c=FrangiFilter3D_crexp(Is,o1); V2c=FrangiFilter3D_crexp(Is,o2);
fid=fopen(out_file,'w'); fwrite(fid,I,'uint16'); fwrite(fid,V1,'single'); fwrite(fid,V1c,'single'); fwrite(fid,V2c,'single'); fclose(fid);
end
