function FeatureiDFT = featiDFTl(fs,num,ESig)

%
%
%
%
% Written by Jiayuan He
% email: jiauyuan.he@uwaterloo.ca
%
%
% for use of this code please cite:
% He, J., Zhang, D., Sheng, X., Li, S., & Zhu, X. (2014). Invariant surface 
% EMG feature against varying contraction level for myoelectric control based
% on muscle coordination. IEEE journal of biomedical and health informatics, 
% 19(3), 874-882.


% tic;

smp = size(ESig,3);
Chl = size(ESig,1);
        
fl = 0.01;
fh = 0.225;

temp = [20 92 163 235 307 378 ]; %均布
temp1 = [92 163 235 307 378 450];

%temp = [20 56 92 128 163 199 235 271 307 343 378 414]; %十二等分
%temp1 = [56 92 128 163 199 235 271 307 343 378 414 450];

sn = size(ESig,2); 
for i = 1:num,
  nf(i,1) = floor(temp(i)*(sn/2)/(fs/2));
  nf(i,2) = floor(temp1(i)*(sn/2)/(fs/2));
end;
if(nf(1,1)==0)
    nf(1,1) = 1;
end;

idft = 0;

for i = 1:smp, %fix samples

    for k = 1:Chl,
%     for k = 3:3,
        sample = ESig(k,:,i);
        temp = fft(sample);
        tl = fix(length(temp)/2);
        temp1 = abs(temp(1:tl));

            for l1 = 1:num,                 
                y = 0;
                y1 = 0;

                for a = 1:(nf(l1,2)-nf(l1,1)),
                    y = y + abs(temp1(a-1+nf(l1,1)));
                end    

%                 y = power(y,2/3);  % change power compression method
                 y = log(y);

                idft = [idft,y];
            end;
    end
    idft = idft(2:size(idft,2));

    FeatureiDFT(i,:) = idft;   
    idft = 0;          
    
end;
             

% toc;        