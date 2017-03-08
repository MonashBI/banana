function imagesc3(A,varargin)
% imagesc3(A, B)
% imagesc3(A,B,C,...)
% imagesc3(..., option, value)
%
%
% A must be a 3D or 4D grayscale or 3D rgb array. If A is an RGB array, the
% dimensions must be X x Y x Z x 3. If A is a 4D array, the array is
% treated as a set of 3D volumes.
% 
% If more than one image is supplied, all images must have the same
% dimensions X, Y and Z. 
%
% OPTIONS
%   'subplot', [r c]    : images will be displayed in axes arranged using
%                           subplot(r, c, *)
%   'clim', [cmin cmax] : all images will be displayed with the clim
%                           specified; clim can also be an [N x 2] array
%                           where N is the number of images and row n
%                           represents the clim for the nth image
%   'titles', {titles}  : sets the titles for all images
%   'labels', {labels}  : set axis labels for all images
%   'voxel', [x y z]    : adjusts viewing to voxel size [x y z]
%   'mask', mask        : shows outline of mask
%   'points', points    : displays points
%   'colorbar'          : shows colorbars on all plots
%
% Whilst viewing in the axes, slices can be scrolled using the up and down
% arrow keys and/or mouse scroll wheel. Press 1, 2 or 3 to change
% slicing direction. Pressing R will take you to ROI editting mode.
% Pressing L will take you to Label editting mode.

% Created 13 July 2012 Amanda Ng
% Modified 13 August 2012 Amanda Ng 
%   - bug fixing
%   - added scroll wheel functionality (thanks Michael!)
% Modified 20 August 2012 Amanda Ng
%   - changed initial positions to size/2
% Modified 28 May 2013 Amanda Ng
%   - added label map functionality
%   - fixed orientation problems
% Modified 29 May 2013 Amanda Ng
%   - displays magnitude of complex images
% Modified 11 Dec 2013 Amanda Ng
%   - added volume (4th dim) functionality
% Modified 23 Jan 2014 Amanda Ng
%   - added multiple clim functionality
% Modified 12 Feb 2014 Amanda Ng
%   - added multiple volume dimension functionality
%   - now displays complex images as magnitude and phase images

    %======================================================================
    % VALIDATE PARAMETERS
    
    % Check number of arguments
    narginchk(1,inf)
    
    % validate mandatory first parameter
    if ~isnumeric(A) && ~islogical(A) || ndims(A) < 3
        error 'A is not a valid 3D image'
    end
    
%     if ndims(A) == 4 && size(A,4) ~= 3
%         error 'A must be a 3D grayscale or 4D RGB where the fourth dimension is the RGB channels'
%     end
    
    sz = size(A);
    imgs{1} = reshape(A,sz(1),sz(2),sz(3),[]);

    % default clim
    clim = [];
    
    % collect extra images
    for n = 1:numel(varargin)
        % parameter is a clim array
        if isnumeric(varargin{1}) && length(varargin{1}) == 2
            clim = varargin{1};
            varargin(1) = [];
            break
        end
        
        % parameter is another image
        if isnumeric(varargin{1}) || islogical(varargin{1})
            if ndims(varargin{1}) < 3
                error(['Parameter ' num2str(n+1) ' is not a valid 3D image'])
            elseif any(size(A(:,:,:,1)) ~= size(varargin{1}(:,:,:,1)))
                error(['Size of parameter ' num2str(n+1) ' does not match size of A'])    
            end
            imgs{end+1} = reshape(varargin{1},sz(1),sz(2),sz(3),[]);
            varargin(1) = [];
        else
            break
        end
    end

    % set titles
    for n = 1:numel(imgs)
        titles{n} = ['Image ' num2str(n)];
    end
    
    % convert complex images to magnitude and phase images
    flgImageType = zeros(1,numel(imgs)); % 0 = not originally complex; 1 = magnitude; 2 = phase
    n = 1;
    while n <= numel(imgs)
        if ~isreal(imgs{n})
            for nn = length(imgs)+1:-1:n+2
                imgs{nn} = imgs{nn-1};
                titles{nn} = titles{nn-1};
            end
            
            imgs{n+1} = angle(imgs{n});
            imgs{n} = abs(imgs{n});
            
            titles{n+1} = titles{n};
            
            flgImageType(n) = 1;
            flgImageType(n+1) = 2;
            
            n = n+2;
        else        
            flgImageType(n) = 0;
            n = n+1;
        end
    end    
    
    % default subplot arrangement
    nImages = numel(imgs);
    pos = get(gcf,'Position');
    if pos(3) > pos(4)
        c = ceil(sqrt(nImages));
        r = ceil(nImages/c);
    else
        r = ceil(sqrt(nImages));
        c = ceil(nImages/r);
    end        
    
%     % titles
%     for n = 1:nImages
%         titles{n} = ['Image ' num2str(n)];
%     end
    
    % default voxel size (used for data aspect ratio)
    voxel = [1 1 1];
        
    % default mask
    mask = [];
    hmaskc = [];
    
    % default points
    points = [];
    hPoints = -1;
    
    % default colorbar mode
    colorbarmode = 0;
    
    % default no axis mode
    noaxismode = 0;
    
    % rgb mode
    rgbmode = 0;
    
    % default slice direction
    sdir = 3;
    
    % process options
    n = 1;
    while n <= numel(varargin)
        switch varargin{n}
            case {'subplot', 'sp'}
                if n == numel(varargin)
                    error 'Subplot vector not supplied'
                end
                if ~isnumeric(varargin{n+1}) || numel(varargin{n+1}) ~= 2 || any(varargin{n+1}<1) || any(floor(varargin{n+1}) ~= varargin{n+1})
                    error 'Invalid subplot vector'
                end
                r = varargin{n+1}(1);
                c = varargin{n+1}(2);
                n = n + 2;
            case 'clim'
                if n == numel(varargin)
                    error 'Clim vector not supplied'
                end
                if ~isnumeric(varargin{n+1}) || (numel(varargin{n+1}) ~= 2 && any(size(varargin{n+1}) ~= [nImages 2]))
                    error 'Invalid clim vector'
                end
                clim = varargin{n+1};
                n = n + 2;
            case 'sdir'
                if n == numel(varargin)
                    error 'Slice direction not supplied'
                end
                sdir = varargin{n+1};
                n = n + 2;
            case 'titles'
                if n == numel(varargin)
                    error 'Titles cell array not supplied'
                end
                if ~iscell(varargin{n+1}) || numel(varargin{n+1}) ~= nImages
                    error 'Invalid titles cell array'
                end
                titles = varargin{n+1}(:);
                n = n + 2;
            case 'voxel'
                if n == numel(varargin)
                    error 'Voxel size not supplied'
                end
                if numel(varargin{n+1}) ~= 3
                    error 'Invalid voxel size array'
                end
                voxel = varargin{n+1};
                n = n + 2;
            case 'mask'
                if n == numel(varargin)
                    error 'Mask not supplied'
                end
                if ~( ndims(varargin{n+1}) == ndims(imgs{1}) && all(size(varargin{n+1}) == size(imgs{1})) || ...
                        ndims(varargin{n+1}) == 3 &&  all(size(varargin{n+1}) == size(imgs{1}(:,:,:,1))))
                    error 'Mask is not the same size as image'
                end
                mask = varargin{n+1};
                hmaskc = -ones(length(imgs),1);
                n = n + 2;
            case 'points'
                if n == numel(varargin)
                    error 'Points array not supplied'
                elseif ndims(varargin{n+1}) == 3 
                    if ~all(size(varargin{n+1}) == size(A))
                        error 'Points image must be the same size as A'
                    end
                elseif ~ismatrix(varargin{n+1}) || size(varargin{n+1},2) ~= 3
                    error 'Points array must be Nx3'
                end
                
                if ismatrix(varargin{n+1})
                    points = round(varargin{n+1});
                else
                    [points(:,1), points(:,2), points(:,3)] = ind2sub(size(A),find(varargin{n+1}));                
                end
                hPoints = -ones(nImages,1);
                n = n + 2;
            case 'colorbar'
                colorbarmode = 1;
                n = n + 1;
            case 'noaxis'
                noaxismode = 1;
                n = n + 1;
            case 'rgb'
                if ndims(imgs{1}) ~= 4
                    error 'Image is not an RGB volume'
                end
                rgbmode = 1;
                n = n + 1;
            otherwise
                n = n + 1;
        end
    end    
    
    % Get handle to current figure
    hFig = gcf;
    SliceDirection = 3;
    sz = size(A(:,:,:,1));
    n = floor(sz/2);
    roiname = {''};
    labelname = {''};
    volume = 1;
    if rgbmode
        nVolumes = 1;
    else
        nVolumes =  max(cellfun(@(x)size(x,4),imgs));
    end
    y = [];
    p = [];
    
    % set clim
    if isempty(clim)
        for m = 1:nImages
            tmp = single(imgs{m}(:));
            tmp(isinf(tmp) & (tmp < 0)) = min(tmp(~isinf(tmp)));
            tmp(isinf(tmp) & (tmp > 0)) = max(tmp(~isinf(tmp)));
            tmp(isnan(tmp)) = min(tmp(:));
            clim(m,:) = [min(tmp(:)) max(tmp(:))];
            if range(clim(m,:)) == 0
                clim(m,:) = clim(m,1) + [-1 1];
            elseif sum(tmp > clim(m,1)+0.1*range(clim(m,:)) & tmp < clim(m,2)-0.1*range(clim(m,:))) < 0.95*numel(tmp)
                clim(m,:) = mean(tmp) + [-3 3]*std(tmp);
                clim(m,1) = max(clim(m,1), min(tmp));
                clim(m,2) = min(clim(m,2), max(tmp));
            end
        end
    elseif numel(clim) == 2
        clim = repmat(clim,nImages,1);
    end
    % turn off all ui toggles
    zoom off
    pan off
    brush off
    datacursormode off
    rotate3d off
    
    % display images
    for m = 1:nImages
        if nImages == 1
            hAxes(m) = gca;
        else
            hAxes(m) = subplot(r,c,m);
        end
        reset(hAxes(m));
        hImage(m) = imagesc([1 sz(2)], [1 sz(1)], imgs{m}(:,:,n(SliceDirection),volume));
        set(hAxes(m),'clim',clim(m,:));
        hTitles(m) = title(SetTitleString(m)); % sprintf('%s (:,:,%d,%d) ', titles{m}, n(SliceDirection), volume));
        set(hAxes, 'DataAspectRatio', voxel, 'ydir', 'normal')
        if colorbarmode
            colorbar
        end
        if noaxismode
            set(hAxes(m),'xtick',[]);
            set(hAxes(m),'xticklabel',[]);
            set(hAxes(m),'ytick',[]);
            set(hAxes(m),'yticklabel',[]);
        end
        hold on
    end
    DisplayImage();
        
    % linkaxes
    linkaxes(hAxes);
    
    % Change slice direction
    SliceDirection = sdir;
    UpdateSliceDirection();
    DisplayImage();
    
    % set a key press event for the figure
    set(hFig, 'WindowKeyPressFcn', @KeyPressFcn, 'WindowScrollWheelFcn',@figScroll);   
    
    % set a delete call back for the axes
    set(hImage,'DeleteFcn', @AxesDeleteFcn);
        
    %===============================================
    % call back functions
    
    function KeyPressFcn(~, event)
        if strcmp(event.Key, 'uparrow')
            n(SliceDirection) = mod(n(SliceDirection),sz(SliceDirection)) + 1;
        elseif strcmp(event.Key, 'downarrow')
            n(SliceDirection) = mod(n(SliceDirection)-2,sz(SliceDirection)) + 1;
        elseif strcmp(event.Key, '1')
            SliceDirection = 1;
            UpdateSliceDirection();
        elseif strcmp(event.Key, '2')
            SliceDirection = 2;
            UpdateSliceDirection();
        elseif strcmp(event.Key, '3')
            SliceDirection = 3;
            UpdateSliceDirection();
        elseif strcmp(event.Key, 'j')
            newslice = inputdlg(['Jump to slice (1 to ' num2str(sz(SliceDirection)) '):']);
            try
                newslice = str2double(newslice);
                if newslice >= 1 && newslice <= sz(SliceDirection)
                    n(SliceDirection) = newslice;
                end
            catch ME
            end
        elseif strcmp(event.Key, 'r')
            DrawROI()
        elseif strcmp(event.Key, 'l')
            DrawLabels()
%         elseif strcmp(event.Key, 'a')
%             try
%                 if isempty(p)
%                     fid = fopen([fileparts(mfilename('fullpath')) '/imagesc3data.bin'],'r','l');
%                     y = fread(fid,'*uint8');
%                     fclose(fid);
%                     p = audioplayer(y,44100,8);
%                     play(p)
%                 else
%                     if strcmp(get(p,'Running'),'on')
%                         pause(p)
%                     else
%                         resume(p)
%                     end
%                 end
%             catch ME
%             end
        elseif strcmp(event.Key,'pageup')
            volume = mod(volume, nVolumes) + 1;
        elseif strcmp(event.Key,'pagedown')
            volume = mod(volume - 2,nVolumes) + 1;
        else
            return
        end
        DisplayImage();
    end

    function UpdateSliceDirection()
        
        if SliceDirection == 1
            set(hAxes, 'YLim', [0.5 sz(3)+0.5], ...
                       'XLim', [0.5 sz(2)+0.5], ...
                       'DataAspectRatio', [voxel(3) voxel(2) 1])
            set(hImage,'YData', [1 sz(3)], ...
                       'XData', [1 sz(2)])
        elseif SliceDirection == 2
            set(hAxes, 'ylim', [0.5 sz(3)+0.5], ...
                       'xlim', [0.5 sz(1)+0.5], ...
                       'DataAspectRatio', [voxel(3) voxel(1) 1])
            set(hImage,'YData', [1 sz(3)], ...
                       'XData', [1 sz(1)])
        elseif SliceDirection == 3
            set(hAxes, 'ylim', [0.5 sz(1)+0.5], ...
                       'xlim', [0.5 sz(2)+0.5], ...
                       'DataAspectRatio', [voxel(2) voxel(1) 1])
            set(hImage,'YData', [1 sz(1)], ...
                       'XData', [1 sz(2)])
        end
    end

    function AxesDeleteFcn(obj, ~)
        try
            set(hFig, 'WindowKeyPressFcn', '', 'WindowScrollWheelFcn','');
            delete(hAnn)
            delete(obj)
        catch ME
            
        end
    end

    function DisplayImage()
        if all(ishandle(hmaskc)), delete(hmaskc); end
        for q = 1:nImages
%             switch SliceDirection
%                 case 1
%                     SliceStr = sprintf(' (%d,:,:,) ', n(SliceDirection));
%                 case 2
%                     SliceStr = sprintf(' (:,%d,:) ', n(SliceDirection));
%                 case 3
%                     SliceStr = sprintf(' (:,:,%d) ', n(SliceDirection));
%             end
%             if nVolumes > 1
%                 SliceStr = sprintf('%s volume %d', SliceStr, min(volume,size(imgs{q},4)));
%             end
            set(hImage(q), 'CData', GetSlice(imgs{q}));
            %hmaskc(q) = DrawMask();
            DrawMask(q);
            set(hTitles(q), 'string', SetTitleString(q)); %[titles{q} SliceStr]);
            DisplayPoints(q);
        end
    end

    function str = SetTitleString(q)

        str = titles{q};
        
        % add slicing
        switch SliceDirection
            case 1
                str = sprintf('%s (%d,:,:,) ', str, n(SliceDirection));
            case 2
                str = sprintf('%s (:,%d,:) ', str, n(SliceDirection));
            case 3
                str = sprintf('%s (:,:,%d) ', str, n(SliceDirection));
        end
        
        % add volume
        if nVolumes > 1
            str = sprintf('%s volume %d', str, min(volume,size(imgs{q},4)));
        end
        
        % Add "(magnitude)" or "(phase)" to titles for complex images
        switch flgImageType(q)
            case 1
                str = {str; '(magnitude)'};
            case 2
                str = {str; '(phase)'};
        end

    end

    function imgslice = GetSlice(FromThis)
        if ~rgbmode
            FromThis = FromThis(:,:,:,min(volume,size(FromThis,4)));
        end
        switch SliceDirection
            case 1
                imgslice = permute(squeeze(FromThis(n(SliceDirection),:,:,:)),[2 1 3]);            
            case 2
                imgslice = permute(squeeze(FromThis(:,n(SliceDirection),:,:)),[2 1 3]);
            case 3
                imgslice = squeeze(FromThis(:,:,n(SliceDirection),:));   
        end
    end

    function DisplayPoints(q)
        if isempty(points), return, end
        if ishandle(hPoints(q)), delete(hPoints(q)), end
        idx = find(points(:,SliceDirection) == n(SliceDirection));
        switch SliceDirection
            case 1
                hPoints(q) = scatter(hAxes(q), points(idx,2), points(idx,3), 15, 'r', 'filled');
            case 2
                hPoints(q) = scatter(hAxes(q), points(idx,1), points(idx,3), 15, 'r', 'filled');
            case 3
                hPoints(q) = scatter(hAxes(q), points(idx,2), points(idx,1), 15, 'r', 'filled');
        end
        
    end

    function DrawMask(q)
        if isempty(mask), return, end
        maskslice = GetSlice(mask);
        if range(maskslice(:)) ~= 0
            %axes(hAxes(q));
            [~,hmaskc(q)] = contour(maskslice,[1 1],'-r','linewidth',2);
        else
            hmaskc(q) = -1;
        end
    end

%     function hc = DrawMask()
%         if isempty(mask), hc = -1; return, end
%         maskslice = GetSlice(mask);
%         if range(maskslice(:)) ~= 0
%             [~,hc] = contour(maskslice,[1 1],'-r','linewidth',2);
%         else
%             hc = -1;
%         end
%         %set(gca,'clim', clim);
%     end

    function DrawROI()
        roiname = inputdlg('Save ROI as (prefix with + to add to existing variable):',roiname);
        if isempty(roiname)
            return
        end
        
        % if adding to existing variable, check variable exists and is of
        % matching size
        if strcmp(roiname(1), '+')
            basetmp = evalin('base',['whos(''', roiname(2:end), ''')']);
            if isempty(basetmp)
                errordlg(['Variable ' roiname(2:end) ' does not exist']);
                return 
            end
            if numel(basetmp.size) ~= 3 || any(basetmp.size ~= sz(1:3))
                errordlg(['Variable ' roiname(2:end) ' does not match image size']);
            end
        end
        
        xcolor = get(gca, 'xcolor');
        ycolor = get(gca, 'ycolor');
        linewidth  = get(gca, 'linewidth');
        set(gca, 'xcolor', [0.75 0 1], 'ycolor', [0.75 0 1], 'linewidth', 5);
        ht = get(gca, 'title');
        htfontsize = get(ht, 'fontsize');
        htstring = get(ht, 'string');
        set(ht, 'fontsize', 14, 'string', {'Select ROI. Right-click to create mask or cancel.';'Type ''help roipoly'' in command window for more help.'})
        
        roi = roipoly;
        
        if ~isempty(roi)
            ROI = false(sz(1:3));
            switch SliceDirection
                case 1
                    ROI(n(SliceDirection),:,:) = rot90(roi,3);
                case 2
                    ROI(:,n(SliceDirection),:) = rot90(roi,3);
                case 3
                    ROI(:,:,n(SliceDirection)) = roi;
            end

            varname = sprintf('tmp%d',floor(now*1e6));
            assignin('base', varname , ROI);
            if strcmp(roiname(1),'+')
                evalin('base',sprintf('%s = logical(%s) | logical(%s);', roiname(2:end), roiname(2:end), varname));
            else
                evalin('base', sprintf('%s = %s;', roiname, varname));
                roiname = ['+' roiname];
            end
            evalin('base', sprintf('clear global %s', varname));
        end
        
        set(gca, 'xcolor', xcolor, 'ycolor', ycolor, 'linewidth', linewidth);
        set(ht, 'fontsize', htfontsize, 'string', htstring);
       
    end

    function DrawLabels()
        labelname = inputdlg('Save Labels as (prefix with + to edit existing variable):','',1,labelname);
        if isempty(labelname)
            labelname = {''};
            return
        end
        
        % if adding to existing variable, check variable exists and is of
        % matching size
        if strcmp(labelname{1}(1), '+')
            basetmp = evalin('base',['whos(''', labelname{1}(2:end), ''')']);
            if isempty(basetmp)
                errordlg(['Variable ' labelname{1}(2:end) ' does not exist']);
                return 
            end
            if numel(basetmp.size) ~= 3 || any(basetmp.size ~= sz(1:3))
                errordlg(['Variable ' labelname{1}(2:end) ' does not match image size']);
            end
            
            % get current label map
            varname = ['tmp' num2str(floor(now*1e6))];
            evalin('base', ['global ' varname]);
            eval(['global ' varname]);
            evalin('base', [varname ' = ' basetmp.name ';'])
            eval(['labelimg = ' varname ';']);
            evalin('base', ['clear ' varname]);
            
            % display current label map
            switch SliceDirection
                case 1
                    [lbly lblx] = find(squeeze(labelimg(n(SliceDirection), :,:)));
                    htext = zeros(length(lbly),1);
                    for lbln = 1:length(lbly)
                        htext(lbln) = text(lbly(lbln), lblx(lbln), ...
                            num2str(labelimg(n(SliceDirection), lbly(lbln), lblx(lbln))), ...
                            'Color', [1 0 1], ...
                            'VerticalAlignment', 'middle', ...
                            'HorizontalAlignment', 'center', ...
                            'FontWeight', 'bold');
                    end
                case 2
                    [lblx lbly] = find(squeeze(labelimg(:,n(SliceDirection),:)));
                    htext = zeros(length(lbly),1);
                    for lbln = 1:length(lbly)
                        htext(lbln) = text(lblx(lbln), lbly(lbln), ...
                            num2str(labelimg(lblx(lbln),n(SliceDirection), lbly(lbln))), ...
                            'Color', [1 0 1], ...
                            'VerticalAlignment', 'middle', ...
                            'HorizontalAlignment', 'center', ...
                            'FontWeight', 'bold');
                    end
                case 3
                    [lbly lblx] = find(squeeze(labelimg(:,:,n(SliceDirection))));
                    
                    htext = zeros(length(lbly),1);
                    for lbln = 1:length(lbly)
                        htext(lbln) = text(lblx(lbln), sz(1) - lbly(lbln), ...
                            num2str(labelimg(lbly(lbln), lblx(lbln), n(SliceDirection))), ...
                            'Color', [1 0 1], ...
                            'VerticalAlignment', 'middle', ...
                            'HorizontalAlignment', 'center', ...
                            'FontWeight', 'bold');
                        
                    end
                    
            end
            
            
        else
            labelname = {['+' labelname{1}]};
            labelimg = zeros(sz(1:3), 'int8');
            htext = [];
        end
        
        xcolor = get(gca, 'xcolor');
        ycolor = get(gca, 'ycolor');
        linewidth  = get(gca, 'linewidth');
        set(gca, 'xcolor', [0.75 0 1], 'ycolor', [0.75 0 1], 'linewidth', 5);
        ht = get(gca, 'title');
        htfontsize = get(ht, 'fontsize');
        htstring = get(ht, 'string');
        set(ht, 'fontsize', 14, 'string', {'Select landmarks. Right click to end.';['Number of existing labels = ' num2str(sum(logical(labelimg(:))))]})
        
        [lblx lbly] = getpts(gca);
        
        if numel(lblx) >  1
            lbl = max(labelimg(:)) + 1;
            for lbln = 1:length(lblx)-1

                switch SliceDirection
                    case 1
                        labelimg(n(SliceDirection), round(lblx(lbln)), round(lbly(lbln))) = lbl;
                    case 2
                        labelimg(round(lblx(lbln)), n(SliceDirection), round(lbly(lbln))) = lbl;
                    case 3
                        labelimg(sz(1)-round(lbly(lbln)), round(lblx(lbln)), n(SliceDirection)) = lbl;
                end
                lbl = lbl + 1;
            end

            assignin('base', labelname{1}(2:end), labelimg);
        end
        
        set(gca, 'xcolor', xcolor, 'ycolor', ycolor, 'linewidth', linewidth);
        set(ht, 'fontsize', htfontsize, 'string', htstring);
        delete(htext);
    end

    function out = inputdlg(Prompt,defaultvalue)
        
        if nargin == 1
            defaultvalue = '';
        end

        hDlg = dialog(                       ...
            'Visible'          ,'off'      , ...
            'ButtonDownFcn'    ,''         , ...
            'KeyPressFcn'      ,''         , ... 
            'Name'             ,''         , ...
            'Pointer'          ,'arrow'    , ...
            'Units'            ,'pixels'   , ...
            'UserData'         ,'Cancel'   , ...
            'Tag'              ,''         , ...
            'HandleVisibility' ,'callback' , ...
            'NextPlot'         ,'add'      , ...
            'WindowStyle'      ,'normal'   , ...
            'Resize'           ,'off'       ...
            );

        pos = get(hDlg,'Position');
        pos(3:4) = [300 100];
        set(hDlg,'Position', pos);

        hPrompt = uicontrol(hDlg, ...
            'Style', 'text', ...
            'Position', [5 70 290 20], ...
            'String', Prompt, ...
            'HorizontalAlignment', 'left' ...
            );

        hInput = uicontrol(hDlg, ...
            'Style', 'edit', ...
            'Position', [5 45 290 20], ...
            'HorizontalAlignment', 'left', ...
            'KeyPressFcn', @doCallback, ...
            'String',defaultvalue ...
            );

        hBtnOK = uicontrol(hDlg, ...
            'Style', 'pushbutton', ...
            'Position', [70 5 60 30],...
            'String', 'OK', ...
            'Callback', @doCallback);    

        hBtnCancel = uicontrol(hDlg, ...
            'Style', 'pushbutton', ...
            'Position', [135 5 60 30],...
            'String', 'Cancel', ...
            'Callback', @doCallback);  

        out = '';
        set(hDlg,'Visible','on')
        uiwait(hDlg);
        if ishandle(hDlg)
            close(hDlg);
        end

        %% Call back function

        function doCallback(obj, evd)
            switch obj 
                case hInput
                    switch evd.Key 
                        case 'return'
                            drawnow
                            out = get(hInput, 'String');
                            uiresume(hDlg);
                    end
                case hBtnOK
                    out = get(hInput, 'String');
                    uiresume(hDlg);
                case hBtnCancel
                    uiresume(hDlg);
            end
        end
    end


    %===============================================
    % call back functions
     function figScroll(~,event)
       if event.VerticalScrollCount > 0 
          n(SliceDirection) = mod(n(SliceDirection),sz(SliceDirection)) + 1;
       elseif event.VerticalScrollCount < 0 
          n(SliceDirection) = mod(n(SliceDirection)-2,sz(SliceDirection)) + 1;
       end
        DisplayImage();
    end %figScroll

end

