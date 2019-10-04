function outputImage = FrangiFilterAutoNoIO( inputImage, maskImage, veinsAreBlack )
%FRANGIFILTERAUTO Frangi3D vessel filter with automatic parameter selection
%   Alpha and Beta set to 0.5
%   Gamma calculated as half the Hessian norm
%   Assumes anisotropic 1 x 1 x 2 voxels

    [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = Hessian3D(inputImage,2);
    hesMatrix = [Dxx(:) Dxy(:) Dxz(:) -Dxy(:) Dyy(:) Dyz(:) -Dxz(:) -Dyz(:) Dzz(:)]';
    hesMatrix = reshape(hesMatrix,[3 3 size(inputImage(:))]);
    hesNorm = zeros(size(inputImage(:)));
    for k=1:size(inputImage(:))
        hesNorm(k) = norm(hesMatrix(:,:,k));
    end   

    options.FrangiScaleRange = [0.25 1.25];
    options.FrangiScaleRatio = 0.25;
    options.FrangiAlpha = 0.5;
    options.FrangiBeta = 0.5;
    options.FrangiC = max(hesNorm)/2;
    options.BlackWhite = veinsAreBlack;
    
    IOut = FrangiFilter3D(inputImage,options);
    thresh = graythresh(IOut(maskImage));
    outputImage = zeros(size(maskImage));
    outputImage(maskImage==1) = im2bw(IOut(maskImage(:)),thresh);
    outputImage(maskImage==0) = 0;
end

