function SoundRecordingtest(duration)
% Basic test and demo of the sound recording/capture functionality of the
% new PsychSound() function.
%
% Parameters: duration = Recordingcapacity (in seconds) of the sound capture
% buffer. After recording of sound has been started, PsychSound will
% perform continuous recording into an internal ring-buffer with a storage capacity
% of 'duration' seconds, until recording gets stopped again. Soundrecording
% is done asynchronously as an automatic background process, independent of
% Matlabs/Psychtoolboxs execution. This way you can continue execution of
% your Matlab script (sound output, response collection, visual output,
% ...) while PsychSound is recording in the background. You can query the
% amount of new recorded sound data in the buffer and you can fetch this
% data from the buffer into a Matlab matrix at any time during of after
% recording. Be aware though that if the buffers capacity is exhausted, it
% will dispose old sound data to store new one - only the last 'duration'
% seconds of the most recently recorded sound are kept.
%
% This leaves you with two options:
%
% a) Request a buffer big enough to store the full duration of your
% recording from Start of capture until the end, and read out the recorded
% data into a Matlab matrix after the end of the recording. This is the
% most easy to implement (nice linear control flow Start->Stop->Fetch), but
% you have to know the maximum duration in advance, and you'll potentially
% use up significant amounts of memory.
%
% b) Request a buffer big enough to hold sound data for at least the
% duration of one iteration of your trial loop. During each iteration, poll
% and fetch portions of the recording into your Matlab matrix. More
% involved programming, but low memory consumption.
%
% With b) you can obviously also implement an audio recorder for infinite
% capture into a file via while(1) Waitabit; Poll&Fetch bit of sound data
% from PsychSound into Matrix; Append content of matrix to a file; end;
%
% And you can do live capture, manipulation & analysis of sound, e.g.,
% voice triggers, feedback of recorded sound to sound output and such.
%

try
   AssertOSX;
catch
	error('This demo does not work under M$-Windows yet, only on MacOS-X. Aborting...');   
end


% Initialize PsychSound for audio recording: We select a sampling frequency
% of 44100 Hz, (which would be also the default if left away) and a capture
% buffer size of 5 seconds. The 'recorderHandle' has the same function as
% the windowPtr for Screen()...
fprintf('Preparing for sound capture...\n');
recorderHandle = PsychSound('InitRecording', 44100, 5);

% Start asynchronous audio recording into capture buffer:
fprintf('Starting continous sound capture. Press a key to stop recording...\n');

while(1)
audiotrigger(0.5, recorderHandle);
[tonset tonset2 sounddata]=audiotrigger(0.5, recorderHandle);
tonset
tonset2
Snd('Play', sounddata, 44100);
WaitSecs(1);
end;

% Shut down sound recorder:
fprintf('Shutting down sound device...\n');
PsychSound('ShutdownRecording', recorderHandle);

return;


tstart = GetSecs;
PsychSound('StartRecording', recorderHandle);
startdelay = GetSecs - tstart

% We wait until keyboard press, while sound is recorded into PsychSounds
% internal buffer.
while(1)
    if KbCheck;
        break;
    end;
    sounddata = PsychSound('GetData', recorderHandle, 1024, 1, 0.001);
    if (max(sounddata)>0.5)
        break;
    end;
    %nrsamples = PsychSound('GetRecordingPosition', recorderHandle)
end;

tonset=GetSecs - tstart



% We stop the recording. The last 5 recorded seconds will be available for retrieval
tend=GetSecs;
PsychSound('StopRecording', recorderHandle);
enddelay=GetSecs - tend

fprintf('Recording stopped...\n');

% Ok, just for fun lets check how much sound data is available for
% retrieval. This could be 5 seconds, if recording lasted at least five
% seconds, so the buffer is full. Could be less if key was pressed earlier
% than 5 seconds. The returned value is the number of audio samples, aka
% nrsamples = availableamount_in_seconds * recordingfrequency = up to
% 5*44100 samples.
nrsamples = PsychSound('GetRecordingPosition', recorderHandle);
fprintf('%i samples have been recorded...\n', nrsamples);

% Now lets fetch everything that's available and return it as a Matlab
% matrix.
ttranss=GetSecs;
%sounddata = PsychSound('GetData', recorderHandle);
transferdelay=GetSecs - ttranss

nrsamples2 = size(sounddata, 1);

fprintf('%i samples have been returned to Matlab...\n', nrsamples2);

% Plot the soundwave as time-series:
plot(sounddata);
drawnow;

% Shut down sound recorder:
fprintf('Shutting down sound device...\n');
PsychSound('ShutdownRecording', recorderHandle);

if 1
    % Play the sound through good'ol SND command:
    for i=1:1
        fprintf('Playback!\n');
        Snd('Play', sounddata, 44100);
        if KbCheck
            break;
        end;
    end;
end;

% Done.
fprintf('Finished. Bye!\n');

return;




function [tonset , tonset2, sounddata] = audiotrigger(threshold, recorderHandle)
PsychSound('StartRecording', recorderHandle);
tstart = GetSecs;
startdelay = GetSecs - tstart;
samplecount=0;

% Wait for keyboard press or sound onset:
while(1)
    if KbCheck;
        break;
    end;
    
    sounddata = PsychSound('GetData', recorderHandle, 1024, 1, 0.001);
    tonset=GetSecs - tstart;
    if (max(sounddata)>threshold)
        break;
    else
        samplecount=samplecount + 1024;
    end;
end;
%samplecount
%min(find(sounddata > threshold))

tonset2=(samplecount + min(find(sounddata > threshold)))/44100;

% We stop the recording. The last 5 recorded seconds will be available for retrieval
tend=GetSecs;
PsychSound('StopRecording', recorderHandle);
%enddelay=GetSecs - tend

return;



