function [recon,cmap]=adapt_array_3d(yn,rn,norm)

% Reconstruction of array data and computation of coil sensitivities based 
% on: a) Adaptive Reconstruction of MRI array data, Walsh et al. Magn Reson
% Med. 2000; 43(5):682-90 and b) Griswold et al. ISMRM 2002: 2410
%-------------------------------------------------------------------------
%	Input:
%	yn: array data (image domain) to be combined [ny, nx, nz, nc]. 
%	rn: data covariance matrix [nc, nc].
%	norm: =1, normalize image intensity
%
%	Output:
%	recon: reconstructed image [ny, nx, nz].
%	cmap: estimated coil sensitivity maps [ny, nx, nz, nc].
%--------------------------------------------------------------------------
% Ricardo Otazo
% CBI, New York University
%
% Modified on 09 Feb 2018 by: Extended to extract 3D maps
% Kamlesh Pawar 
% Monash University, 
% Extend to extract 3D maps
%--------------------------------------------------------------------------
%

yn=permute(yn,[4,1,2,3]);
[nc,nx,ny,nz]=size(yn);
if nargin<3, norm=0; end
if nargin<2, rn=eye(nc);end

% find coil with maximum intensity for phase correction
[mm,maxcoil]=max(sum(sum(sum(permute(abs(yn),[4 3 2 1])))));   

bs1=8;  %x-block size
bs2=8;  %y-block size
bs3=8;  %z-block size
st=4;   %increase to set interpolation step size

% wsmall=zeros(nc,round(nx./st),round(ny./st),round(nz./st));
cmapsmall=zeros(nc,round(nx./st),round(ny./st),round(nz./st));
for z = st:st:nz
    fprintf('processing partition: %d\n',z);
    for y = st:st:ny
        for x = st:st:nx
            %Collect block for calculation of blockwise values
            zmin1=max([z-bs3./2 1]);
            ymin1=max([y-bs2./2 1]);                   
            xmin1=max([x-bs1./2 1]);                  
            % Cropping edges
            zmax1=min([z+bs3./2 nz]);                 
            ymax1=min([y+bs2./2 ny]);   
            xmax1=min([x+bs1./2 nx]);   

            lz1=length(zmin1:zmax1);
            ly1=length(ymin1:ymax1);
            lx1=length(xmin1:xmax1);
            m1=reshape(yn(:,xmin1:xmax1,ymin1:ymax1,zmin1:zmax1),nc,lx1*ly1*lz1);

            m=m1*m1'; %signal covariance

            % eignevector with max eigenvalue for optimal combination
            [e,v]=eig(inv(rn)*m);                    

            v=diag(v);
            [mv,ind]=max(v);

            mf=e(:,ind);                      
            mf=mf/(mf'*inv(rn)*mf);               
            normmf=e(:,ind);

            % Phase correction based on coil with max intensity
            mf=mf.*exp(-1i*angle(mf(maxcoil)));        
            normmf=normmf.*exp(-1i*angle(normmf(maxcoil)));

%             wsmall(:,x./st,y./st,z./st)=mf;
            cmapsmall(:,x./st,y./st,z./st)=normmf;
        end
    end
end

% Interpolation of weights upto the full resolution
% Done separately for magnitude and phase in order to avoid 0 magnitude 
% pixels between +1 and -1 pixels.
% wfull = zeros(nc,nx,ny,nz);
cmap = zeros(nc,nx,ny,nz);
for i=1:nc
    fprintf('Processing Channel: %d\n', i);
%     wfull(i,:,:,:)=conj(imresize3D(squeeze(abs(wsmall(i,:,:,:))),[nx ny nz],'bilinear').*exp(1i.*imresize3D(angle(squeeze(wsmall(i,:,:,:))),[nx ny nz],'nearest')));
    cmap(i,:,:,:)=imresize3D(squeeze(abs(cmapsmall(i,:,:,:))),[nx ny nz],'bilinear').*exp(1i.*imresize3D(squeeze(angle(cmapsmall(i,:,:,:))),[nx ny nz],'nearest'));
end
% recon=squeeze(sum(wfull.*yn));   %Combine coil signals. 
recon=squeeze(sum(conj(cmap).*yn));   %Combine coil signals. 
% clear wfull
% normalization proposed in the abstract by Griswold et al.
if norm
    recon=recon.*squeeze(sum(abs(cmap))).^2; 
end

cmap=squeeze(cmap);
end

function [imOut] = imresize3D(im,dims,method)
% IMRESIZE3D 
% This function resizes 3D image
% 
% [OUTPUTARGS] = IMRESIZE3D(INPUTARGS) Explain usage here
% 
% Examples: 
% 
% Provide sample usage code here
% 
% See also: List related files here

% Author: Kamlesh Pawar 
% Date: 2018/02/09 15:58:04 
% Revision: 0.1 $
% Institute: Monash Biomedical Imaging, Monash University, Australia, 2018

[y,x,z]=...
   ndgrid(linspace(1,size(im,1),dims(1)),...
          linspace(1,size(im,2),dims(2)),...
          linspace(1,size(im,3),dims(3)));
imOut=interp3(im,x,y,z, method);
end