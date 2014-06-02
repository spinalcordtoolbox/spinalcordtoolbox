% =========================================================================
% FUNCTION
% j_createParadigm.m
%
% COMMENTS
% julien cohen-adad 2007-02-23
% =========================================================================
function paradigm = j_createParadigm()


% initialization
file_write          = 'sequence_block.txt'; % put '' for no file output
nb_stimuli          = [6 3]; % number of stimuli per block. There can be several stimuli. THE OUTPUT FILE WILL ONLY CONTAIN THE FIRST STIMULUS TYPE
value_stimuli       = [1];
random_scheme       = 1; % stimuli (if more than one) are sequenced randomly. If 0, then stimuli type are put one after another
time_inter_stimuli  = [18]; % in second. If two values [a b] are given, then inter-stimuli duration is randomly distributed around a +/- b
nb_blocks           = 4; % number of blocks per session
time_inter_block    = 0; % in second
time_start          = 0; % in second. First period of rest

% bloc_length         = 1; % in second

timing              = 'relative'; % relative, absolute
unit                = 's'; % s, ms
add_fictive_stim    = 1; % for Sinai software, one must add a fictive stimulation long time after the last bloc. Otherwise it loops on the sequence.

% misc
tr                  = 2; % in second
% display_seq         = 1; % plot the paradigm
% sequence_duration   = 600; % in second



% build all blocks of stimuli
nb_total_stimuli = sum(nb_stimuli);
nb_stimuli_type = size(nb_stimuli,2);
for i_stimuli_type=1:nb_stimuli_type
    sequence_relative(i_stimuli_type).onsets = [];
    first_block(i_stimuli_type) = 1;
end
adjust_positionning = zeros(1,nb_stimuli_type);
for i_block=1:nb_blocks
    
    % build stimuli array
    stimuli = [];
    for i_stimuli_type=1:nb_stimuli_type
        stimuli = cat(2,stimuli,value_stimuli(i_stimuli_type)*ones(1,nb_stimuli(i_stimuli_type)));
    end

    % randomize stimuli
    if (random_scheme)
        random_sequence = rand(1,nb_total_stimuli);
        [random_sequence_sorted index_stimuli] = sort(random_sequence);
        stimuli = stimuli(index_stimuli);
    end

    % put stimuli in time together
    for i_stimuli_type=1:nb_stimuli_type
        stimuli_relative(i_stimuli_type).value = value_stimuli(i_stimuli_type);
        stimuli_relative(i_stimuli_type).onsets = [];
        i_count = 1;
        first_stimulus = 1;
        for i_stimuli=1:nb_total_stimuli
            if stimuli(i_stimuli)==value_stimuli(i_stimuli_type)
                stimuli_relative(i_stimuli_type).onsets = cat(2,stimuli_relative(i_stimuli_type).onsets,time_inter_stimuli*i_count + first_stimulus*(time_inter_block+adjust_positionning(i_stimuli_type)*time_inter_stimuli) + first_block(i_stimuli_type)*time_start);
                i_count = 1;
                first_stimulus = 0;
                first_block(i_stimuli_type) = 0;
            else
                i_count = i_count + 1;
            end
        end
    end
    
    % concatenate stimuli generated after last block
    for i_stimuli_type=1:nb_stimuli_type
        sequence_relative(i_stimuli_type).value = value_stimuli(i_stimuli_type);
        sequence_relative(i_stimuli_type).onsets = cat(2,sequence_relative(i_stimuli_type).onsets,stimuli_relative(i_stimuli_type).onsets);
    end
    
    % adjust relative positionning for next block to come (this code is really a mess...)
    for i_stimuli_type=1:nb_stimuli_type
        adjust_positionning(i_stimuli_type) = nb_total_stimuli - max(find(stimuli==value_stimuli(i_stimuli_type)));
    end
end

% build absolute sequence
for i_stimuli_type=1:nb_stimuli_type
    sequence_absolute(i_stimuli_type).value = sequence_relative(i_stimuli_type).value;
    for i_stimuli=1:length(sequence_relative(i_stimuli_type).onsets)
        sequence_absolute(i_stimuli_type).onsets(i_stimuli) = sum(sequence_relative(i_stimuli_type).onsets(1:i_stimuli));
    end
end

% display sequence
t_max = max(cat(2,sequence_absolute(1:end).onsets))+time_inter_stimuli;
t=(0:1:t_max/tr);
for i_stimuli_type=1:nb_stimuli_type
    sequence(i_stimuli_type,:) = zeros(1,t_max/tr+1);
    for i=1:t_max/tr
        if find(tr*t(i)==sequence_absolute(i_stimuli_type).onsets)
            sequence(i_stimuli_type,i) = sequence_absolute(i_stimuli_type).value;
        end    
    end
end
figure; plot(tr*t,sequence,'linewidth',2);
hold on
plot(tr*t,zeros(1,length(t)),'k','linewidth',2);
xlabel('time (s)')
ylabel('onset')
ylim([-2 2])
grid

% adjust relative/absolute timing
switch timing
    case 'relative'
    sequence_output = sequence_relative(1).onsets;
    case 'absolute'
    sequence_output = sequence_absolute(1).onsets;
end

% add fictive stimulus
if (add_fictive_stim)
    sequence_output = cat(2,sequence_relative(1).onsets,3600);
end

% convert in specified unit
switch unit
    case 'ms'
    sequence_output = sequence_output*1000;
end

% write file
if ~isempty(file_write)
    fid = fopen(file_write,'w');
    fprintf(fid,'%i\n',sequence_output);
    fclose(fid);
end

% output
paradigm = sequence_output;





% ==========================================
%       OLD CODE
% ==========================================
% clear
% 
% nb_blocs = 10;
% file_source = 'eventFlash-003.par';
% file_name = 'eventFlash-serge003.txt';
% 
% % open file
% f_file=fopen(file_name,'w+');
% 
% if file_source
%     % read data
%     fid=fopen(file_source);
%     a=fscanf(fid,'%f %i %f %*s',[3 inf]);
%     a=a(1,2:2:end)'*1000;
%     fclose(fid);
%     % write data
%     for i=1:size(a,1)
%         fprintf(f_file,'%d\n',a(i));
%     end
% else
%     for iBloc=1:nb_blocs
%         for iEvent=1:2:20
%             fprintf(f_file,'%d\n',(20*(2*iBloc-1)+iEvent-1)*1000);
%         end
%     end
% end
% 
% % close file 
% fclose(f_file);


% % generate sequence
% clear sequence
% onsets = [];
% if (absolute_timing)
%     for i=1:nb_blocs
%         onsets = cat(2,onsets,first_bloc+(i-1)*(2*bloc_length)+(0:tr:bloc_length-tr));
%     end
% else
%     % first bloc
%     onsets = cat(2,onsets,first_bloc);
%     onsets = cat(2,onsets,ones(1,(bloc_length/tr)-1)*tr);
%     % following blocs
%     for i=1:nb_blocs-1
%         onsets = cat(2,onsets,bloc_length+tr,ones(1,(bloc_length/tr)-1)*tr);
%     end
% end





% if display_seq
%     t = (0:tr:sequence_duration-1);
%     sequence = zeros(1,sequence_duration/tr);
% %     run_length  = nb_blocs*(bloc_length*2);
%     i_onset = 1;
%     for i=1:sequence_duration/tr-1
%         if i_onset==length(onsets_absolute)+1
%             break
%         end
%         if t(i)==onsets_absolute(i_onset)
%             sequence(i) = 1;
%             i_onset = i_onset + 1;
%         end
%     end
%     figure; plot(t,sequence,'r*');
%     xlabel('time (s)')
%     ylabel('onset')
%     ylim([0 1.5])
%     grid
% end




