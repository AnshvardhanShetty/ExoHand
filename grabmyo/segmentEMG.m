function [ESig  ELabel] = segmentEMG(Sig,WTime,STime,MTime,SFreq,MLabel)
% To segment signal by a certain rule to a cell, which is 1*n that n is for
% the number of motions. Every unit is a A * B *C matrix that A stands for 
% the number of data in an analysis window, B stands for the number of
% the channel, C stands for the number of samples in one motion.
% Caution: this function did not compute the transition time between
% motions. So it fits to compute the training data.
% Input variables:
%                Sig is original signal.                              data by channel matrix 
%                WTime is the time of an analysis window.             seconds
%                STime is the time of a sliding window.               seconds
%                MTime is the lasting time of a single motion.        seconds
%                SFreq is the Sample Frequency.
%                MLabel is the label of the motions.   vector
%
% Written by Jiayuan He
% email: jiauyuan.he@uwaterloo.ca

ELabel = 0;
winLen = floor(WTime * SFreq);
sldLen = floor(STime * SFreq);
monLen = floor(MTime * SFreq);
% winLen = WTime * SFreq;
% sldLen = STime * SFreq;
% monLen = MTime * SFreq;
k1 = size(MLabel,1);   % the times of all the motions
k2 = floor((monLen - winLen) / sldLen);  % the times of window sliding in one motion time,not computing the transition of motions

% 剔除动作转换时的数据
% ----------------- segment signal ---------------------
for i = 1: k1,  % fix the motion    
    for j = 0: k2,  % fix the sliding window
          ESig(:,:,(i-1)*(k2+1)+j+1) = Sig(round(((i-1) * monLen + j * sldLen + 1)):round(((i-1) * monLen + j * sldLen + winLen)),:)';
    end
    temp = repmat(MLabel(i),k2+1,1);
    ELabel = [ELabel;temp];
end
ELabel = ELabel(2:size(ELabel,1),:);

%%
%Sample
% for i = 1:10
% [D{i},L{i}] = SegmentEMG(D7((i-1)*27000+1:i*27000,:),0.2,0.025,3,1000,La);
% end;
% tsD6 = cat(3,D{1:i});
% tsL6 = cat(1,L{1:i});



    