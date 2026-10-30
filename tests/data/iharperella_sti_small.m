% Writes tests/data/iharperella_sti_small.bin: STI Suite 3.0's iHARPERELLA on one small phantom
% with three settings, for tests/iharperella_sti_parity.rs.
%
% Recovered 2026-10-30 from the session that made the fixture (2026-10-21, Bunya), unchanged
% except for the path and these comments. Run as a script in licensed MATLAB R2023b:
%   addpath('<STISuite_V3.0>/Core_Functions_P'); iharperella_sti_small
% No QSM-CI container runs iHARPERELLA yet (QSMxT/QSM-CI#300, part B3); once one is verified, this
% fixture is to be made with it, so it can be regenerated in Actions.
%
% Layout (little-endian), per case: the number of mask voxels (u32), then iHARPERELLA's tissue
% phase at the mask voxels (f64, column-major).
N=[32 32 24]; [x,y,z]=ndgrid(1:N(1),1:N(2),1:N(3));
mask=double(((x-16.5).^2+(y-16.5).^2+((z-12.5)*1.2).^2)<100);
f=0.3*sin(0.4*x+0.2*y).*cos(0.3*z)+0.002*x.*y;
cases={ {[1 1 1.5],[4 5 6],5}, {[1 1 1],[0 0 0],8}, {[0.8 0.9 1.2],[3 3 3],40} };
fid=fopen('iharperella_sti_small.bin','w');
for c=1:numel(cases)
  C=cases{c}; tp=iHARPERELLA(f,mask,'voxelsize',C{1},'padsize',C{2},'niter',C{3});
  v=tp(mask>0); fwrite(fid,uint32(numel(v)),'uint32'); fwrite(fid,v,'double');
end
fclose(fid);
