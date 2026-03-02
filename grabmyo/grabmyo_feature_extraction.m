% This script will process the converted .mat using fileread.m
% to your hard drive. The converted files are located in 'Output BM' and
% organzied in 3 folders 'Session 1_converted','Session 2_converted' and
% 'Session 3_converted'. Each folder contains 43 .mat files, with two cell
% matrices DATA_FOREARM and DATA_WRIST
%
% DATA_FOREARM 7x17 cell matrix (trials x gestures)
% Each cell: 10240x16 (sampfreq*5sec x wristchannels)
%
% % DATA_WRIST 7x17 cell matrix (trials x gestures)
% Each cell: 10240x12 (sampfreq*5sec x wristchannels)
%
% Standard windowing of EMG signal is performed followed by frequency
% division technique feature extraction
% In order to run this script make sure 'Output BM' and srgmentEMG and
% featIDFT.m are in the same directory
%
% output %%%%%%%%%
% Main Folder: 'Feature Extracted BM'
% File: 'Forearm_Session1.mat' 'Forearm_Session2.mat' 'Forearm_Session3.mat'
% VarOut: FeatSet (NSUB x NGESTURES cell matrix)
% Each cell: NSAMPLE x NFEATURES
%
%
% Forearm Electrode Configuration %%%%%%%%%
%  1  2  3  4  5  6  7  8
%  9 10 11 12 13 14 15 16
%
% Wrist Electrode Configuration %%%%%%%%%
%  1  2  3  4  5  6
%  7  8  9 10 11 12
%
% Written by Ashirbad Pradhan
% email: ashirbad.pradhan@uwaterloo.ca

clear
addpath(genpath([pwd filesep 'Output BM']))

%obtain the total number of subject. Note the total subjects should be same
% in all the three folders
fs = 2048;                   %sampling frequency
NSUB = length(dir([pwd filesep 'Output BM' filesep 'Session1_converted']))-2; %no of participants
NSESSION = length(dir([pwd filesep 'Output BM']))-2;
NGESTURE = 16;              %total number of gestures
NTRIALS = 7;                %total number of trials
a1=[];                      % temporary variable to merge trials of a specific contraction
a2=[];                      % temporary variable to
CompleteSet=[];        % store final gestures and subjects for each session

%% Define output folder
if ~exist('Feature Extracted BM', 'dir')
    mkdir('Feature Extracted BM')
else
    disp('Overwriting')
    rmdir('Feature Extracted BM','s')
    mkdir('Feature Extracted BM')
end

%% flatten the trials to obtain Gestures x Subjects
for isession=1:NSESSION
    for isub =1:NSUB
        fileName = ['session' num2str(isession) '_participant',num2str(isub),'.mat'];
        temp_load_forearm=load(fileName,'DATA_FOREARM');
        datafile=temp_load_forearm.DATA_FOREARM;
        for igesture = 1:NGESTURE+1        % +1 to include rest gesture
            for itrial=1:NTRIALS
                a1=[a1; datafile{itrial,igesture}];
            end
            a2=[a2,{a1}];
            a1=[];
        end
        CompleteSet=[CompleteSet;a2];
        a2=[];
        disp(['Loaded: ' num2str(isession) ' ' num2str(isub)])
    end
end
rmpath(genpath([pwd filesep 'Output BM']))         % to save memory

%% segmentation and processing
count=0;
for isession = 1: NSESSION
    FeatSet={};
    for isub =1:NSUB
        for igesture = 1:NGESTURE
            OneSet = CompleteSet{(isession-1)*NSUB+isub,igesture}';       %shape=16xTotalSamples
            post_process=[];
            for ichannel_2=1:8         % monopolar 8 channel with average referencing
                temp2 = OneSet(1:8,:);
                post_process(ichannel_2,:)=OneSet(ichannel_2,:)-mean(temp2,1);
            end
            % %         for ichannel_2=1:8         % bipolar 8 channel processing
            % %             post_process(ichannel_2,:)=OneSet(ichannel_2,:)-OneSet(ichannel_2+8,:);
            % %         end
            % segment the EMG Data
            segData = segmentEMG(post_process', 0.2, 0.15, NTRIALS*5, fs, 1);  % post_process' has to be NSampx8
            %extract the frequency features
            feat= featiDFTl(2048,6,segData);
            FeatSet(isub,igesture)={feat};
        end
        count=count+1;
        disp(['processed: ',num2str(count),' of ', num2str(NSUB*NSESSION),' files'])
    end
    disp(['saving: Session ',num2str(isession),' biometric data'])
    save(['Feature Extracted BM' filesep 'Forearm_Session' num2str(isession) '.mat'],'FeatSet')
end