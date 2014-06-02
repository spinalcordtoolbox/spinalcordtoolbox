clear

% load tous les fichiers
load sica
card=j_readFile('card_s1r1.txt');
for i=1:30
    tc(:,i)=sica.A(:,i);
end

% normalise moyenne et ecart type pour tc
for i=1:30
    tc(:,i)=tc(:,i)-mean(tc(:,i));
    tc(:,i)=tc(:,i)/std(tc(:,i));
end

% normalise moyenne et ecart type pour card
cardM=card-mean(card);
cardM=cardM(1:1557);
cardM=cardM';
cardM=cardM/std(cardM);
clear card

% calcule l'autocorrélation
for i=1:30
%     r(:,i)=corr2(cardM,tc(:,i));
    rx(:,i)=xcorr(cardM,tc(:,i));
end

% trouve le max
for i=1:30
    max_rx(i)=max(abs(rx(:,i)));
end

% ordonne
% [rs rs_ind]=sort(abs(r),'descend');
[rsx rsx_ind]=sort(abs(max_rx),'descend');


% figure; plot(tc(:,2))
% hold on
% plot(cardM,'r')

% rx_test(:,4)=xcorr(cardM,tc(:,1),'none');

figure; plot(abs(rx(:,1)))
hold on
plot(conv_cardMtc1,'r')

% figure; plot(r(:,1:30))
% xlim([790 810])
