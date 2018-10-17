function FrangiFilterAuto( inputFile, maskFile, outputFile, veinsAreBlack )
%FRANGIFILTERAUTO Frangi3D vessel filter with automatic parameter selection
%   Alpha and Beta set to 0.5
%   Gamma calculated as half the Hessian norm
%   Assumes anisotropic 1 x 1 x 2 voxels

    inputImage = load_nii(inputFile);
    fileHeader = inputImage.hdr;
    inputImage = single(inputImage.img);
            
    [xi,yi,zi] = meshgrid(1:1:size(inputImage,2), 1:1:size(inputImage,1), 1:0.5:size(inputImage,3));
    isoInputImage = interp3(inputImage,xi,yi,zi);
    [xf,yf,zf] = meshgrid(1:1:size(isoInputImage,2), 1:1:size(isoInputImage,1), 1:2:size(isoInputImage,3));
    
    maskImage = load_nii(maskFile);
    maskImage = single(maskImage.img);
    isoMask = interp3(maskImage,xi,yi,zi)>0;
    
    [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = Hessian3D(isoInputImage,2);
    hesMatrix = [Dxx(:) Dxy(:) Dxz(:) -Dxy(:) Dyy(:) Dyz(:) -Dxz(:) -Dyz(:) Dzz(:)]';
    hesMatrix = reshape(hesMatrix,[3 3 size(isoInputImage(:))]);
    hesNorm = zeros(size(isoInputImage(:)));
    for k=1:size(isoInputImage(:))
        hesNorm(k) = norm(hesMatrix(:,:,k));
    end   

    options.FrangiScaleRange = [0.25 1.25];
    options.FrangiScaleRatio = 0.25;
    options.FrangiAlpha = 0.5;
    options.FrangiBeta = 0.5;
    options.FrangiC = max(hesNorm)/2;
    options.BlackWhite = veinsAreBlack;
    
    IOut = FrangiFilter3D(isoInputImage,options);
    thresh = graythresh(IOut(isoMask));
    venogram = zeros(size(isoMask));
    venogram(isoMask==1) = im2bw(IOut(isoMask(:)),thresh);
    venogram(isoMask==0) = 0;

    venogram = interp3(venogram,xf,yf,zf);
                
    outputImage = make_nii(venogram);
    outputImage.hdr = fileHeader;
    save_nii(outputImage, outputFile);

end

