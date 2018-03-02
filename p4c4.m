%2017-12-16 after e-mailing teacher ,decide to build a new version using
%the paper's procedure.(TBME_2010)
% This program predicts the first character in session 12, run 01, using a very simple classification method
% This classification method uses only one sample (at 310ms) and one channel (Cz) for classification
% 
% (C) Gerwin Schalk; Dec 2002
clc;clear;
fprintf(1, '2nd Wadsworth Dataset for Data Competition:\n');
fprintf(1, 'Data from a P300-Spelling Paradigm\n');
fprintf(1, '-------------------------------------------\n');
fprintf(1, '(C) Gerwin Schalk 2002\n\n');

% load data file
fprintf(1, 'Loading data file for session 12, run 01\n');
files = { 'AAS010R01' , 'AAS010R02', 'AAS010R03' , 'AAS010R04'...
    'AAS010R05' , 'AAS011R01' , 'AAS011R02' , 'AAS011R03', ...
    'AAS011R04' , 'AAS011R05' , 'AAS011R06' , 'AAS012R01' , ...
    'AAS012R02' , 'AAS012R03' , 'AAS012R04' , 'AAS012R05' , ...
    'AAS012R06' , 'AAS012R07' , 'AAS012R08' };
load( files{ 1 } );

samplefreq=240;
triallength=round(600*samplefreq/1000);     % samples in one evoked response 一个激励响应信号的采样数
max_stimuluscode=12;
titlechar='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789-';
channelselect=[ 3 4 7:26 28:32 34:36 ];          %11, 18, 51, 58, 62, 53, 4, 10, 12
                                                 %34, 4, 11, 18, 9, 13, 51, 58, 62, 47, 49, ... 53, 55, 56, 57, 59, 60, 61, 63
                                                 %around 9(C3), 13(C4), 11(Cz),  18(CPz), 4(FCz), except 47(P7), 55(P8)

% 0.1-20Hz bandpass filter on the signal
signal = passband_filter(signal);

% get a list of the samples that divide one character from the other
idx=find(PhaseInSequence == 3);                                % get all samples where PhaseInSequence == 3 (end period after one character)
charsamples=idx(find(PhaseInSequence(idx(1:end-1)+1) == 1));   % get exactly the samples at which the trials end (i.e., the ones where the next value of PhaseInSequence equals 1 (starting period of next character))
if ( trialnr(charsamples(1))==0)
    charsamples(1)=[];
end
% this determines the first and last intensification to be used here
% in this example, this results in evaluation of intensification 1...180 (180 = 15 sequences x 12 stimuli)
starttrial=min(trialnr)+1;                                     % intensification to start is the first intensification
endtrial=max(trialnr);        % the last intensification is the last one for the first character

stimulusdata=zeros(endtrial, triallength);
stimulusy=zeros(endtrial,1);

% go through all intensifications and calculate classification results after each
fprintf(1, 'Going through all intensifications for the first character\n');
for cur_trial=starttrial:endtrial
 % get the indeces of the samples of the current intensification
 trialidx=find(trialnr == cur_trial);
 % get the data for these samples (i.e., starting at time of stimulation and triallength samples
 trialdata=signal(min(trialidx):min(trialidx)+triallength-1, :);
 stimulusdata(cur_trial, :)=mean((trialdata(:,channelselect))');
 stimulusy(cur_trial,:)=StimulusType(min(trialidx))+1;
end % session

oddidx = find(stimulusy == 2);
disoddidx = find(stimulusy == 1);
stimulusdata = featureNormalize(stimulusdata);%归一化
disoddidxRandSelect = disoddidx(randperm(length(disoddidx),1*length(oddidx)));%从0样本中选取适量训练样本，零一样本比为2.

train_data = stimulusdata([oddidx;disoddidxRandSelect],:);%训练数据
train_y = stimulusy([oddidx;disoddidxRandSelect],:);

stimulusdata = train_data;
stimulusy = train_y;

%% Feature Normalization
%% Setup the parameters you will use for this part of the exercise
num_labels = 2;          % 2 labels, from 1 to 2
                          % (note that we have mapped "0" to label 1 and "1" to label 2)
n = size(stimulusdata, 2);
all_theta = zeros(num_labels, n + 1);
m = size(stimulusdata, 1);
%% ============ Part 2a: Training Parameters ============
lambda = 0.01;
[all_theta] = oneVsAllValid(stimulusdata, stimulusy, num_labels, lambda, all_theta);

%% ============ Valid ===================================
load( files{ 2 } );
signal = passband_filter(signal);

% get a list of the samples that divide one character from the other
idx=find(PhaseInSequence == 3);                                % get all samples where PhaseInSequence == 3 (end period after one character)
charsamples=idx(find(PhaseInSequence(idx(1:end-1)+1) == 1));   % get exactly the samples at which the trials end (i.e., the ones where the next value of PhaseInSequence equals 1 (starting period of next character))
if ( trialnr(charsamples(1))==0)
    charsamples(1)=[];
end
% this determines the first and last intensification to be used here
% in this example, this results in evaluation of intensification 1...180 (180 = 15 sequences x 12 stimuli)
starttrial=min(trialnr)+1;                                     % intensification to start is the first intensification
endtrial=max(trialnr);        % the last intensification is the last one for the first character

stimulusdata=zeros(endtrial, triallength);
stimulusy=zeros(endtrial,1);

% go through all intensifications and calculate classification results after each
fprintf(1, 'Going through all intensifications for the first character\n');
for cur_trial=starttrial:endtrial
 % get the indeces of the samples of the current intensification
 trialidx=find(trialnr == cur_trial);
 % get the data for these samples (i.e., starting at time of stimulation and triallength samples
 trialdata=signal(min(trialidx):min(trialidx)+triallength-1, :);
 stimulusdata(cur_trial, :)=mean((trialdata(:,channelselect))');
 stimulusy(cur_trial,:)=StimulusType(min(trialidx))+1;
end % session

stimulusdata = featureNormalize(stimulusdata);
oddidx = find(stimulusy == 2);
disoddidx = find(stimulusy == 1);
stimulusdata = featureNormalize(stimulusdata);%归一化
disoddidxRandSelect = disoddidx(randperm(length(disoddidx),3*length(oddidx)));%从0样本中选取适量训练样本，零一样本比为2.

train_data = stimulusdata([oddidx;disoddidxRandSelect],:);%训练数据
train_y = stimulusy([oddidx;disoddidxRandSelect],:);

stimulusdata = train_data;
stimulusy = train_y;

[all_theta] = oneVsAllValid(stimulusdata, stimulusy, num_labels, lambda, all_theta);
% pred = predictOneVsAll(all_theta, stimulusdata);
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred' == stimulusy)) * 100);

%% ================== Test =====================
load( files{ 3 } );
signal = passband_filter(signal);

% get a list of the samples that divide one character from the other
idx=find(PhaseInSequence == 3);                                % get all samples where PhaseInSequence == 3 (end period after one character)
charsamples=idx(find(PhaseInSequence(idx(1:end-1)+1) == 1));   % get exactly the samples at which the trials end (i.e., the ones where the next value of PhaseInSequence equals 1 (starting period of next character))
if ( trialnr(charsamples(1))==0)
    charsamples(1)=[];
end
% this determines the first and last intensification to be used here
% in this example, this results in evaluation of intensification 1...180 (180 = 15 sequences x 12 stimuli)
starttrial=min(trialnr)+1;                                     % intensification to start is the first intensification
endtrial=max(trialnr);        % the last intensification is the last one for the first character

stimulusdata=zeros(endtrial, triallength);
stimulusy=zeros(endtrial,1);

% go through all intensifications and calculate classification results after each
fprintf(1, 'Going through all intensifications for the first character\n');
for cur_trial=starttrial:endtrial
 % get the indeces of the samples of the current intensification
 trialidx=find(trialnr == cur_trial);
 % get the data for these samples (i.e., starting at time of stimulation and triallength samples
 trialdata=signal(min(trialidx):min(trialidx)+triallength-1, :);
 stimulusdata(cur_trial, :)=mean((trialdata(:,channelselect))');
 stimulusy(cur_trial,:)=StimulusType(min(trialidx))+1;
end % session

stimulusdata = featureNormalize(stimulusdata);
pred = (predictOneVsAll(all_theta, stimulusdata))';

pred_correct = find(pred == 2);
correct_label = 0;
for i=1:length(pred_correct)
    if stimulusy(pred_correct(i)) == 2
        correct_label =correct_label+1;
    end
end

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == stimulusy)) * 100);

%% ==========================分字母验证============================
chartrials = trialnr(charsamples);
trialamond =[0;chartrials;max(trialnr)];

    count = 1;
for charnr=1:length(pred_correct)
    if(length(trialamond(pred_correct(charnr)>trialamond)) == 1)
       char1trial(count) = pred_correct(charnr);
       count = count + 1;
    end
end
char1trial = char1trial';
%% =====================确定预测字母==================================
for i=1:length(char1trial)
    char1pro(i)=StimulusCode(min(find(trialnr ==char1trial(i))));
end
    countcol = 1;
    countrow = 1;
for i=1:length(char1pro)
    if(char1pro(i)<7)
        char1column(countcol)=char1pro(i);
        countcol = countcol + 1;
    else
        char1row(countrow)=char1pro(i);
        countrow = countrow + 1;
    end
end

table=tabulate(char1row);
[F,I]=max(table(:,2));
I=find(table(:,2)==F);
result=table(I,1)
        
titlechar((char1row-7)*6+char1column)









