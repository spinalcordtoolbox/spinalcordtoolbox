function SaveMovieFrames(framenr,filmnaam,filmdir,outdir)
% function SaveMovieFrames(framenr,filmname,filmdir,outdir):
%
% loads a GUI in which movie FILMNAME in FILMDIR can be played and seeked
% through. Movie is opened on frame FRAMENR. Screenshots of frames can be
% saved as png in OUTDIR by pressing the save button.

% JJvR 26-02-2008

%% Initialise GUI figure

        figure;
        set(gcf,'Visible','off','Position',[600 300 900 600]);
        set(gca,'Position',[.1 .1 .7 .8]);
        axis ij; axis equal; axis off;
        data    = guidata(gcf);
        data.filmnaam   = filmnaam;
        data.framenr    = framenr;
        data.filmdir    = filmdir;
        data.outdir     = outdir;
        data.filminfo   = aviinfo([data.filmdir filesep data.filmnaam]);
        pause on;
        guidata(gcf,data);

        takeframe;
        plotframe;

        hpluseen = uicontrol(...
            'Style','pushbutton',...
            'String','Next Frame',...
            'Position',[750,500,120,40],...
            'Callback',{@Pluseen_Callback},...
            'Tag','Pluseen');

        hmineen = uicontrol(...
            'Style','pushbutton',...
            'String','Prev Frame',...
            'Position',[750,425,120,40],...
            'Callback',{@Mineen_Callback},...
            'Tag','Mineen');

        hplusvijf = uicontrol(...
            'Style','pushbutton',...
            'String','5 forward',...
            'Position',[750,350,120,40],...
            'Callback',{@Plusvijf_Callback},...
            'Tag','Plusvijf');

        hminvijf = uicontrol(...
            'Style','pushbutton',...
            'String','5 back',...
            'Position',[750,275,120,40],...
            'Callback',{@Minvijf_Callback},...
            'Tag','Minvijf');

        hplay = uicontrol(...
            'Style','togglebutton',...
            'String','Play',...
            'Position',[750,200,120,40],...
            'Callback',{@Play_Callback},...
            'Tag','Minvijf');

        hSave = uicontrol(...
            'Style','pushbutton',...
            'String','Save',...
            'Position',[750,125,120,40],...
            'Callback',{@Save_Callback},...
            'Tag','SaveButton');

        hTerug = uicontrol(...
            'Style','pushbutton',...
            'String','Close',...
            'Position',[750,50,120,40],...
            'Callback',{@Terug_Callback},...
            'Tag','SaveButton');
        
        set(gcf,'Visible','on');
    end
%% Callbacks

    function Pluseen_Callback(source,eventdata,handles)
        updaterelframe(1);
        plotframe;
    end

    function Mineen_Callback(source,eventdata,handles)
        updaterelframe(-1);
        plotframe;
    end

    function Plusvijf_Callback(source,eventdata,handles)
        updaterelframe(5);
        plotframe;
    end

    function Minvijf_Callback(source,eventdata,handles)
        updaterelframe(-5);
        plotframe;
    end
    
    function Play_Callback(source,eventdata,handles)
        data = guidata(gcf);
        playing = true;
        while playing
            if (get(gcbo, 'Value')) == 0
                playing = false;
            end   
            updaterelframe(1);
            pause(1/25);
            plotframe;
        end
    end
    
    function Save_Callback(source,eventdata,handles)
        data = guidata(gcf);
        if ~isdir(data.outdir)
            mkdir(data.outdir);
        end
        imwrite(data.film(data.relframe).cdata,[data.outdir filesep data.filmnaam(1:end-4) ' frame ' num2str(data.framenr - 21 + data.relframe) '.png'],'png');
    end

    function Terug_Callback(source,eventdata,handles)
        close(gcf);
    end
%% Supporting functions

    function takeframe
        data = guidata(gcf);
        data.addrelframe = 0;
        if data.framenr + 40 > data.filminfo.NumFrames
            data.addrelframe = 40 - (data.filminfo.NumFrames - data.framenr);
            data.framenr = data.filminfo.NumFrames - 40; 
            if data.addrelframe > 40
                data.addrelframe = 40;
            end
        elseif data.framenr - 10 < 1
            data.addrelframe = data.framenr - 10;
            data.framenr = 11;
            if data.addrelframe < -10
                data.addrelframe = -10;
            end
        end
        
        data.film = aviread([data.filmdir filesep data.filmnaam], data.framenr-10:data.framenr+40);
        data.relframe = 11 + data.addrelframe;
 
        guidata(gcf,data);
    end

    function plotframe
        data = guidata(gcf);
        image(data.film(data.relframe).cdata);
    end

    function updaterelframe(input)
        data = guidata(gcf);
        data.relframe = data.relframe+input;
        guidata(gcf,data);
        checkrelframe;
    end
    
    function checkrelframe
        data = guidata(gcf);
        if data.relframe > 51  || data.relframe < 1
            getnewrelframe;
        end
    end
    
    function getnewrelframe
        data = guidata(gcf);
        data.framenr = (data.framenr - 11 + data.relframe);
        guidata(gcf,data);
        takeframe;
    end

