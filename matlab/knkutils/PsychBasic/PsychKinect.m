function varargout = PsychKinect(varargin)
% PsychKinect -- Control and access the Microsoft Kinect depth camera.
%
% This is a high level driver to allow convenient access to the Microsoft
% Kinect box. The Kinect is a depth-sensing "3D camera". The Kinect
% consists of a standard color camera (like any standard USB webcam)
% to capture a scene at 640x480 pixels resolution in RGB8 color with up to
% 30 frames per second. In addition it has a depth sensor that measures the
% distance of each "pixel" from the camera. The Kinect delivers a color
% image with depth information which can be used to infer the 3D structure
% of the observed visual scene and to perform a 3D reconstruction of the
% scene.
%
% See KinectDemo and Kinect3DDemo for two demos utilizing PsychKinect to
% demonstrate this basic functions of the Kinect.
%
% PsychKinect internally uses the PsychKinectCore MEX file which actually
% interfaces with the Kinect.
%
% The driver is currently supported on Microsoft Windows under
% Matlab version 7.4 (R2007a) and later. It is also supported on
% GNU/Linux with Matlab or Octave and on Intel based Macintosh computers
% under OS/X with Matlab or Octave. The driver supports all versions of
% the Microsoft Kinect on Linux and OSX, but currently only the original
% XBOX-360 Kinect under Microsoft Windows.
%
% To use this driver you need:
% 1. A Microsoft Kinect (price tag about 150$ at December 2010).
% 2. A interface cable to connect the Kinect to a standard USB port of a
%    computer - sold separately or included in standalone Kinects.
% 3. The free and open-source libfreenect + libusb libraries and drivers
%    from the OpenKinect project (Homepage: http://www.openkinect.org )
%
% You need to install these libraries separately, otherwise our driver will
% abort with an "Invalid MEX file" error.
%
% Type "help InstallKinect" for installation instructions and licensing
% terms.
%
%
% Subfunctions:
%
% Most functions are part of the PsychKinectCore() mex file, so you can get
% help for them by typing PsychKinectCore FUNCTIONNAME ? as usual, with
% FUNCTIONNAME being the name of the function you want to get help for.
%
% PsychKinect('Shutdown');
% - Release all internal resources of PsychKinect.
%
%
% PsychKinect('ApplyCalibrationFile', kinect, calibFileName);
% - Load Kinect calibration from a .yml calibration file 'calibFileName', as
% created by rgbDemo software and apply it to Kinect with handle 'kinect'.
%
%
% kobject = PsychKinect('CreateObject', window, kinect [, oldkobject]);
% - Create a new kobject for the specified 'window', using the Kinect box
% specified by the given 'kinect' handle. Recycle 'oldkobject' so save
% memory and resources if 'oldkobject' is provided. Otherwise create a new
% object.
%
% Do not use within creen('BeginOpenGL', window); and Screen('EndOpenGL',
% window); calls, as 2D mode is needed.
%
%
% kobject encodes the 3D geometry of the scene as sensed by the Kinect. It
% corresponds to the currently selected "3d video frame" from the kinect,
% as selected by PsychKinect('GrabFrame', kinect);
% kobject can then by accessed directly (it is a struct variable) or passed
% to other PsychKinect functions for display and processing.
%
% PsychKinect('DeleteObject', window, kobject);
% - Delete given 'kobject' for given 'window' once you no longer need it.
% During a work-loop you could also pass 'kobject' to the next
% PsychKinect('CreateObject', ...); call as 'oldkobject' to recycle it for
% reasons of computational efficiency.
%
% Do not use within creen('BeginOpenGL', window); and Screen('EndOpenGL',
% window); calls, as 2D mode is needed.
%
%
% PsychKinect('DrawObject', window, kobject [, drawtype=0]);
% Draw the 3D scene stored in 'kobject' into the 'window' selected. Use the
% current OpenGL settings for this. 'drawtype' is optional and defines kind
% of rendering: 0 (the default) draws a colored point-cloud of all sensed
% 3D points. 1 draws a dense textured 3D surface mesh.
%
% This function must be enclosed between Screen('BeginOpenGL', window);
% and Screen('EndOpenGL', window); calls, as 3D mode is needed.
%

% History:
%  5.12.2010  mk   Initial version written.

global GL;

persistent kinect_opmode;
persistent glsl;
persistent idxvbo;
persistent idxbuffersize;
persistent isCalibrated;

% Command specified? Otherwise we output the help text of us and
% the low level driver:
if nargin > 0
    cmd = varargin{1};
else
    help PsychKinect;
    PsychKinectCore;
    return;
end

if isempty(isCalibrated)
    isCalibrated = 0;
end

if strcmpi(cmd, 'Shutdown');
    glsl = [];
    idxvbo = [];
    idxbuffersize = [];
    isCalibrated = 0;
    kinect_opmode = [];
    return;
end

if strcmpi(cmd, 'ApplyCalibrationFile')

    if nargin < 2 || isempty(varargin{2})
        error('You must provide a valid "kinect" handle as 1st argument!');
    end
    kinect = varargin{2};

    if nargin < 3 || isempty(varargin{3}) || ~ischar(varargin{3})
        error('You must provide a valid "calibrationFile" filename as 2nd argument!');
    end
    calFilename = varargin{3};
    [fd errmsg] = fopen(calFilename, 'rt');
    if fd == -1
	error('Could not open calibration file "%s". Error was: %s.\n', calFilename, errmsg);
    end

    if isempty(strfind(fgets(fd), 'YAML'))
	fclose(fd);
	error('Not a valid Kinect calibration file "%s"!\n', calFilename);
    end

    while 1
	cur = fgets(fd);
	if cur == -1
		break;
	end

	if strfind(cur, 'rgb_intrinsics')
		fgets(fd);
		fgets(fd);
		fgets(fd);
		cur = '';
		while isempty(strfind(cur, ']'))
			cur = strcat(cur , fgets(fd));
		end
		sm = strfind(cur, '[') + 1;
		em = strfind(cur, ']') - 1;
		cur = cur(sm:em);
		mat = sscanf(cur, '%f %*s %f %*s %f %*s %f %*s %f %*s %f %*s %f %*s %f %*s %f');
		rgb_intrinsics = transpose(reshape(mat, 3, 3))
	end

	if strfind(cur, 'depth_intrinsics')
		fgets(fd);
		fgets(fd);
		fgets(fd);
		cur = '';
		while isempty(strfind(cur, ']'))
			cur = strcat(cur , fgets(fd));
		end
		sm = strfind(cur, '[') + 1;
		em = strfind(cur, ']') - 1;
		cur = cur(sm:em);
		mat = sscanf(cur, '%f %*s %f %*s %f %*s %f %*s %f %*s %f %*s %f %*s %f %*s %f');
		depth_intrinsics = transpose(reshape(mat, 3, 3))
	end

	if strfind(cur, 'R')
		fgets(fd);
		fgets(fd);
		fgets(fd);
		cur = '';
		while isempty(strfind(cur, ']'))
			cur = strcat(cur , fgets(fd));
		end
		sm = strfind(cur, '[') + 1;
		em = strfind(cur, ']') - 1;
		cur = cur(sm:em);
		mat = sscanf(cur, '%f %*s %f %*s %f %*s %f %*s %f %*s %f %*s %f %*s %f %*s %f');
		R = transpose(reshape(mat, 3, 3))
	end

	if strfind(cur, 'depth_distortion')
		fgets(fd);
		fgets(fd);
		fgets(fd);
		cur = '';
		while isempty(strfind(cur, ']'))
			cur = strcat(cur , fgets(fd));
		end
		sm = strfind(cur, '[') + 1;
		em = strfind(cur, ']') - 1;
		cur = cur(sm:em);
		mat = sscanf(cur, '%f %*s %f %*s %f %*s %f %*s %f');
		depth_distortion = mat
	end

	if strfind(cur, 'rgb_distortion')
		fgets(fd);
		fgets(fd);
		fgets(fd);
		cur = '';
		while isempty(strfind(cur, ']'))
			cur = strcat(cur , fgets(fd));
		end
		sm = strfind(cur, '[') + 1;
		em = strfind(cur, ']') - 1;
		cur = cur(sm:em);
		mat = sscanf(cur, '%f %*s %f %*s %f %*s %f %*s %f');
		rgb_distortion = mat
	end

	if strfind(cur, 'T')
		fgets(fd);
		fgets(fd);
		fgets(fd);
		cur = '';
		while isempty(strfind(cur, ']'))
			cur = strcat(cur , fgets(fd));
		end
		sm = strfind(cur, '[') + 1;
		em = strfind(cur, ']') - 1;
		cur = cur(sm:em);
		mat = sscanf(cur, '%f %*s %f %*s %f');
		T = mat
	end

	if strfind(cur, 'depth_base_and_offset')
		fgets(fd);
		fgets(fd);
		fgets(fd);
		cur = '';
		while isempty(strfind(cur, ']'))
			cur = strcat(cur , fgets(fd));
		end
		sm = strfind(cur, '[') + 1;
		em = strfind(cur, ']') - 1;
		cur = cur(sm:em);
		mat = sscanf(cur, '%f %*s %f');
		depth_base_and_offset = mat
	end
    end

    fclose(fd);

    % Done parsing the file. Apply parameters to specified kinect:
    PsychKinect('SetBaseCalibration', kinect, [depth_intrinsics(1,1), depth_intrinsics(2,2), depth_intrinsics(1,3), depth_intrinsics(2,3)], ...
					      [rgb_intrinsics(1,1), rgb_intrinsics(2,2), rgb_intrinsics(1,3), rgb_intrinsics(2,3)], ...
					      R, T, depth_distortion, rgb_distortion, depth_base_and_offset);
    isCalibrated = 1;
    fprintf('PsychKinect: Info: Calibration from file %s applied to kinect handle %i.\n', calFilename, kinect);

    return;
end

if strcmpi(cmd, 'CreateObject')

    if nargin < 2 || isempty(varargin{2})
        error('You must provide a valid "window" handle as 1st argument!');
    end
    win = varargin{2};

    if nargin < 3 || isempty(varargin{3})
        error('You must provide a valid "kinect" handle as 2nd argument!');
    end
    kinect = varargin{3};

    if nargin < 4 || isempty(varargin{4})
        kmesh.tex = [];
        kmesh.vbo = [];
        kmesh.buffersize = 0;
    else
        kmesh = varargin{4};
    end

    if isempty(kinect_opmode)
	exts = glGetString(GL.EXTENSIONS);
	has_vbos = ~isempty(findstr(exts, '_vertex_buffer_object'));
	has_shader = ~isempty(findstr(exts, 'GL_ARB_shading_language')) && ...
		     ~isempty(findstr(exts, 'GL_ARB_shader_objects')) && ...
		     ~isempty(findstr(exts, 'GL_ARB_vertex_shader')) && ...
		     ~isempty(findstr(exts, 'GL_ARB_fragment_shader'));
	has_gshader = has_shader && ~isempty(findstr(exts, 'GL_ARB_geometry_shader'));
	has_drawinst = ~isempty(findstr(exts, 'GL_ARB_draw_instanced'));

	% Ok, this is not strictly correct in theory, but pretty much in practice;-) 
	has_vtfetch = has_gshader && (glGetIntegerv(GL.MAX_VERTEX_TEXTURE_IMAGE_UNITS) > 0);

	if ~has_vbos
		% Ancient hardware: Use slow cpu path:
		kinect_opmode = 0;
		fprintf('PsychKinect: VBO''s not supported by your GPU: Using slow path for mesh rendering.\n');
	else
		% At least VBO's supported:
		kinect_opmode = 1;
		fprintf('PsychKinect: VBO''s supported by your GPU: Using faster path for mesh rendering.\n');

		% Shader support, so we can do almost all math in the vertex shader?
		if has_shader
			% Could use 2 or 3, but 3 does depths conversion in shader and
			% is drastically faster than 2. Downside: Only float precision
			% instead of double precision for conversion, however no perceptible
			% difference.
			kinect_opmode = 3;
			fprintf('PsychKinect: Shaders supported by your GPU: Using fast shader-path for mesh rendering.\n');
		end

		% Support for geometry shaders, vertex texture fetch and instanced draw?
		if has_shader && has_gshader && has_vtfetch && has_drawinst
			% TODO: IMPLEMENT SUPPORT kinect_opmode = 4;
		end
	end
    end

    kmesh.xyz = [];
    kmesh.rgb = [];

    % Turn RGB video camera image into a Psychtoolbox texture and corresponding
    % OpenGL rectangle texture:
    [imbuff, width, height, channels] = PsychKinectCore('GetImage', kinect, 0, 1);
    if width > 0 && height > 0
        kmesh.tex = Screen('SetOpenGLTextureFromMemPointer', win, kmesh.tex, imbuff, width, height, channels, 1, GL.TEXTURE_RECTANGLE_EXT);
        [ gltexid gltextarget ] =Screen('GetOpenGLTexture', win, kmesh.tex);
        kmesh.gltexid = gltexid;
        kmesh.gltextarget = gltextarget;
    else
        varargout{1} = [];
        fprintf('PsychKinect: WARNING: Failed to fetch RGB image data!\n');
        return;
    end

    if kinect_opmode == 0
        % Dumb mode: Return a complete matrix with encoded vertex positions
        % and vertex colors:
        [foo, width, height, channels, glformat] = PsychKinect('GetDepthImage', kinect, 2, 0);
        foo = reshape (foo, 6, size(foo,2) * size(foo,3));
        kmesh.xyz = foo(1:3, :);
        kmesh.rgb = foo(4:6, :);
        kmesh.type = 0;
        kmesh.glformat = glformat;
    end

    if kinect_opmode == 1
        % Fetch databuffer with preformatted data for a VBO that
        % contains interleaved (vx,vy,vz) 3D vertex positions and (tx,ty)
        % 2D texture coordinates, i.e., (vx,vy,vz,tx,ty) per element:
        [vbobuffer, width, height, channels, glformat] = PsychKinect('GetDepthImage', kinect, 3, 1);
        if width > 0 && height > 0
            Screen('BeginOpenGL', win);
            if isempty(kmesh.vbo)
                kmesh.vbo = glGenBuffers(1);
            end
            glBindBuffer(GL.ARRAY_BUFFER, kmesh.vbo);
            kmesh.buffersize = width * height * channels * 8;
            glBufferData(GL.ARRAY_BUFFER, kmesh.buffersize, vbobuffer, GL.STREAM_DRAW);
            glBindBuffer(GL.ARRAY_BUFFER, 0);
            Screen('EndOpenGL', win);
            kmesh.Stride = channels * 8;
            kmesh.textureOffset = 3 * 8;
            kmesh.nrVertices = width * height;
            kmesh.type = 1;
            kmesh.glformat = glformat;
        else
            varargout{1} = [];
            fprintf('PsychKinect: WARNING: Failed to fetch VBO geometry data!\n');
            return;
        end
    end

    if kinect_opmode == 2 || kinect_opmode == 3
        if isempty(glsl)
            % First time init of shader:

	    % Fetch all camera calibration parameters from PsychKinectCore for this kinect:
	    [depthsIntrinsics, rgbIntrinsics, R, T, depthsUndistort, rgbUndistort, depth_base_and_offset] = PsychKinectCore('SetBaseCalibration', kinect);
	    [fx_d, fy_d, cx_d, cy_d] = deal(depthsIntrinsics(1), depthsIntrinsics(2), depthsIntrinsics(3), depthsIntrinsics(4));
	    [fx_rgb, fy_rgb, cx_rgb, cy_rgb] = deal(rgbIntrinsics(1), rgbIntrinsics(2), rgbIntrinsics(3), rgbIntrinsics(4));
	    [k1_d, k2_d, p1_d, p2_d, k3_d] = deal(depthsUndistort(1), depthsUndistort(2), depthsUndistort(3), depthsUndistort(4), depthsUndistort(5));
	    [k1_rgb, k2_rgb, p1_rgb, p2_rgb, k3_rgb] = deal(rgbUndistort(1), rgbUndistort(2), rgbUndistort(3), rgbUndistort(4), rgbUndistort(5));

            if kinect_opmode == 2
                % Standard shader: Doesn't do initial sensor -> depths conversion.
                glsl = LoadGLSLProgramFromFiles('KinectShaderStandard');
            else
                % Compressed shader: Does this first step as well, albeit only with
                % single precision float precision instead of the double precision of
                % the C implementation. Drastically faster and no perceptible difference,
                % but that doesn't mean there isn't any:
                glsl = LoadGLSLProgramFromFiles('KinectShaderCompressed');
            end

            % Assign all relevant camera parameters to shader: Optical undistortion data isn't
            % used yet, but would be easy to do at least for the rgb camera, within a fragment
            % shader:
            glUseProgram(glsl);
            glUniform4f(glGetUniformLocation(glsl, 'depth_intrinsic'), fx_d, fy_d, cx_d, cy_d);
            glUniform4f(glGetUniformLocation(glsl, 'rgb_intrinsic'), fx_rgb, fy_rgb, cx_rgb, cy_rgb);
            glUniformMatrix3fv(glGetUniformLocation(glsl, 'R'), 1, GL.TRUE, R);
            glUniform3fv(glGetUniformLocation(glsl, 'T'), 1, T);
	    if kinect_opmode ~= 2
		    glUniform2fv(glGetUniformLocation(glsl, 'depth_base_and_offset'), 1, depth_base_and_offset);
	    end
            glUseProgram(0);
            repeatedscan = 0;
        else
            repeatedscan = 1;
        end

        if isempty(idxvbo)
            toponame = [PsychtoolboxConfigDir 'kinect_quadmeshtopology.mat'];
            if exist(toponame,'file')
                load(toponame);
            else
                tic;
                fprintf('Building mesh topology. This is a one time effort that can take some seconds. Please standby...\n');
                % Build static fixed mesh topology for a GL_QUADS style mesh:
                meshindices = uint32(zeros(4, 639, 479));
                for yi=1:479
                    for xi=1:639
                        meshindices(1, xi, yi) = (yi-1+0) * 640 + (xi-1+0);
                        meshindices(2, xi, yi) = (yi-1+0) * 640 + (xi-1+1);
                        meshindices(3, xi, yi) = (yi-1+1) * 640 + (xi-1+1);
                        meshindices(4, xi, yi) = (yi-1+1) * 640 + (xi-1+0);
                    end
                end
                save(toponame, '-V6', 'meshindices');
                fprintf('...done. Saved to file to save startup time next time you use me. Took at total of %f seconds.\n\n', toc);
            end

            idxvbo = glGenBuffers(1);
            glBindBuffer(GL.ELEMENT_ARRAY_BUFFER, idxvbo);
            idxbuffersize = 479 * 639 * 4 * 4;
            glBufferData(GL.ELEMENT_ARRAY_BUFFER, idxbuffersize, meshindices, GL.STATIC_DRAW);
            glBindBuffer(GL.ELEMENT_ARRAY_BUFFER, 0);
        end

        % Fetch databuffer with preformatted data for a VBO that
        % contains interleaved (x,y,vz) 3D vertex positions:
        if kinect_opmode == 2
            format = 4 + repeatedscan;
        else
            % Opmode 3 outsources computation of raw depths from raw sensor data to the
            % Vertex shader as well, maybe with slightly reduced precision:
            format = 6 + repeatedscan;
        end

        [vbobuffer, width, height, channels, glformat] = PsychKinect('GetDepthImage', kinect, format, 1);
        if width > 0 && height > 0
            Screen('BeginOpenGL', win);
            if isempty(kmesh.vbo)
                kmesh.vbo = glGenBuffers(1);
            end
            glBindBuffer(GL.ARRAY_BUFFER, kmesh.vbo);
            kmesh.buffersize = width * height * channels * 8;
            glBufferData(GL.ARRAY_BUFFER, kmesh.buffersize, vbobuffer, GL.STREAM_DRAW);
            glBindBuffer(GL.ARRAY_BUFFER, 0);
            Screen('EndOpenGL', win);
            kmesh.Stride = channels * 8;
            kmesh.textureOffset = 0;
            kmesh.nrVertices = width * height;
            kmesh.type = kinect_opmode;
            kmesh.glsl = glsl;
            kmesh.glformat = glformat;
        else
            varargout{1} = [];
            fprintf('PsychKinect: WARNING: Failed to fetch VBO geometry data!\n');
            return;
        end
    end

    % Store handle to kinect:
    kmesh.kinect = kinect;
    kmesh.idxvbo = idxvbo;
    kmesh.idxbuffersize = idxbuffersize;
    varargout{1} = kmesh;

    return;
end

if strcmpi(cmd, 'DeleteObject')
    if nargin < 2 || isempty(varargin{2})
        error('You must provide a valid "window" handle as 1st argument!');
    end
    win = varargin{2};

    if nargin < 3 || isempty(varargin{3})
        error('You must provide a valid "mesh" struct as 2nd argument!');
    end
    kmesh = varargin{3};

    if ~isempty(kmesh.tex)
        Screen('Close', kmesh.tex);
    end

    if kmesh.type == 0
        return;
    end

    if kmesh.type == 1 || kmesh.type == 2 || kmesh.type == 3
        if ~isempty(kmesh.vbo)
            glDeleteBuffers(1, kmesh.vbo);
        end
        return;
    end

    return;
end

if strcmpi(cmd, 'RenderObject')

    if nargin < 2 || isempty(varargin{2})
        error('You must provide a valid "window" handle as 1st argument!');
    end
    win = varargin{2};

    if nargin < 3 || isempty(varargin{3})
        error('You must provide a valid "mesh" struct as 2nd argument!');
    end
    kmesh = varargin{3};

    if nargin < 4 || isempty(varargin{4})
        drawtype = 0;
    else
        drawtype = varargin{4};
    end

    % Primitive way: All data encoded inside kmesh.xyz, kmesh.rgb as
    % double matrices. Use PTB function to draw it. Sloooow:
    if kmesh.type == 0
        moglDrawDots3D(win, kmesh.xyz, 2, kmesh.rgb, [], 1);
    end

    % VBO with encoded texture coordinates?
    if kmesh.type == 1
        % Yes. No need for GPU post-processing, just bind & draw:

        glPushAttrib(GL.ALL_ATTRIB_BITS);

        % Activate and bind texture on unit 0:
        glActiveTexture(GL.TEXTURE0);
        glEnable(kmesh.gltextarget);
        glBindTexture(kmesh.gltextarget, kmesh.gltexid);

        % Textures color texel values shall modulate the color computed by lighting model:
        glTexEnvfv(GL.TEXTURE_ENV,GL.TEXTURE_ENV_MODE,GL.MODULATE);
        glColor3f(1,1,1);

        % Use alpha blending, so fragment alpha has desired effect:
        glEnable(GL.BLEND);
        glBlendFunc(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA);

        % Activate and bind VBO:
        glEnableClientState(GL.VERTEX_ARRAY);
        glBindBuffer(GL.ARRAY_BUFFER, kmesh.vbo);
        glVertexPointer(3, GL.DOUBLE, kmesh.Stride, 0);
        glEnableClientState(GL.TEXTURE_COORD_ARRAY);
        glTexCoordPointer(2, GL.DOUBLE, kmesh.Stride, kmesh.textureOffset);

        % Pure point cloud rendering requested?
        if drawtype == 0
            glDrawArrays(GL.POINTS, 0, kmesh.nrVertices);
        end

        if drawtype == 1
            glBindBuffer(GL.ELEMENT_ARRAY_BUFFER, kmesh.idxvbo);
            glDrawRangeElements(GL.QUADS, 0, 479 * 640 + 639, kmesh.idxbuffersize / 4, GL.UNSIGNED_INT, 0);
            glBindBuffer(GL.ELEMENT_ARRAY_BUFFER, 0);
        end

        glBindBuffer(GL.ARRAY_BUFFER, 0);
        glDisableClientState(GL.VERTEX_ARRAY);
        glDisableClientState(GL.TEXTURE_COORD_ARRAY);
        glBindTexture(kmesh.gltextarget, 0);
        glDisable(kmesh.gltextarget);
        glPopAttrib;
    end

    if kmesh.type == 2 || kmesh.type == 3
        % Yes. Need for GPU post-processing:

        % Activate and bind texture on unit 0:
        glPushAttrib(GL.ALL_ATTRIB_BITS);
        glActiveTexture(GL.TEXTURE0);
        glEnable(kmesh.gltextarget);
        glBindTexture(kmesh.gltextarget, kmesh.gltexid);

        % Textures color texel values shall modulate the color computed by lighting model:
        glTexEnvfv(GL.TEXTURE_ENV,GL.TEXTURE_ENV_MODE,GL.MODULATE);
        glColor3f(1,1,1);

        % Use alpha blending, so fragment alpha has desired effect:
        glEnable(GL.BLEND);
        glBlendFunc(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA);

        % Activate and bind VBO:
        glEnableClientState(GL.VERTEX_ARRAY);
        glBindBuffer(GL.ARRAY_BUFFER, kmesh.vbo);
        if kmesh.type == 3
            glVertexPointer(2, kmesh.glformat, kmesh.Stride, 0);
        else
            glVertexPointer(3, kmesh.glformat, kmesh.Stride, 0);
        end
        glUseProgram(kmesh.glsl);

        % Pure point cloud rendering requested?
        if drawtype == 0
            glDrawArrays(GL.POINTS, 0, kmesh.nrVertices);
        end

        if drawtype == 1
            glBindBuffer(GL.ELEMENT_ARRAY_BUFFER, kmesh.idxvbo);
            glDrawRangeElements(GL.QUADS, 0, 479 * 640 + 639, kmesh.idxbuffersize / 4, GL.UNSIGNED_INT, 0);
            glBindBuffer(GL.ELEMENT_ARRAY_BUFFER, 0);
        end

        glUseProgram(0);
        glBindBuffer(GL.ARRAY_BUFFER, 0);
        glDisableClientState(GL.VERTEX_ARRAY);
        glBindTexture(kmesh.gltextarget, 0);
        glDisable(kmesh.gltextarget);
        glPopAttrib;
    end

    return;
end

% No matching command found: Pass all arguments to the low-level
% PsychKinectCore mex file driver. Low level command might be
% implemented there:
[ varargout{1:nargout} ] = PsychKinectCore(varargin{:});

return;
