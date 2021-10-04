clc 
clear all 
close all

%% global variables
% Based on sensor configuration.
 %  numADCBits = 16; % number of ADC bits per sample.
   numADCSamples = 151; % number of ADC samples per chirp.
   numRx = 4; % number of receivers in AWR1243.
   chirpSize = numADCSamples*numRx;
   chirploops= 128; % No. of of chirp loops.  
   numLanes = 4; % do not change. number of lanes is always 4.
   isReal = 0; % set to 1 if real only data, 0 if complex data.
   numFrames = 100; 
   frameTime = 0.100;   % 40ms per frame
   totalTime = frameTime*numFrames;
   numChirps = 128;
   sampleRate = 5; % [Msps]
   timeStep = 1/sampleRate;    % [us]
   chirpPeriod = numADCSamples * timeStep ; % [us]
   plotEnd = numADCSamples * numChirps*numFrames; %for considering all frames.
   Dx = numADCSamples * numChirps ;
   timeEnd = (plotEnd-1) * timeStep;

%% read file
% read .bin file
fid = fopen('1.6.bin','r');
adcData = fread(fid, 'int16');
fclose(fid);
fileSize = size(adcData, 1);
%% organize data by LVDS lane
% for complex data
  remaind = mod(fileSize,8);
% Make data(Interleaved Data from AWR1243) over 8 columns.
% if remaind ~= 0 
%    adcData =[ adcData;zeros(8-remaind,1)] ;
% end
% fileSize = length(adcData);

%% stroing data in LVDS if Real and in cmplx if complex(IQ from mmwave studio)   
if isReal % For real data 4 columns for 4 receivers
    adcData = adcData'; 
    LVDS = reshape(adcData ,4,[])';
else
% cmplx has 4 real & 4 imaginary columns for 4 Rceivers for interleaved data format.
    adcData = adcData';
    cmplx = reshape(adcData ,8,[])';
end

%% return receiver data
if isReal 
    retValue = LVDS; 
else
    retValue = cmplx;
end

% plotting the data
adcData = retValue ;

real_rece_1 = adcData(:,1); imag_rece_1 = adcData(:,5);
real_rece_2 = adcData(:,2); imag_rece_2 = adcData(:,6);
real_rece_3 = adcData(:,3); imag_rece_3 = adcData(:,7);
real_rece_4 = adcData(:,4); imag_rece_4 = adcData(:,8);

% % Distance calculation using d=(c*f/(2*slope))
fdel_bin = (0:1:numADCSamples-1)*((5*10^6)/numADCSamples);
slope = 80*10^6;
distance = ((1.5*10^2)*fdel_bin)/slope;
% ff = slope*2*distance/(3*10^8);

%%
% plotting for all frames (Channel 1)
R1 = reshape(real_rece_1,numADCSamples,[])'; I1 = reshape(imag_rece_1,numADCSamples,[])';

%%
R2 = reshape(real_rece_2,numADCSamples,[])'; I2 = reshape(imag_rece_2,numADCSamples,[])';
R3 = reshape(real_rece_3,numADCSamples,[])'; I3 = reshape(imag_rece_3,numADCSamples,[])';
R4 = reshape(real_rece_4,numADCSamples,[])'; I4 = reshape(imag_rece_4,numADCSamples,[])';

complete_signal_fft_1 = []; 
complete_signal_fft_2 = [];
complete_signal_fft_3 = [];
complete_signal_fft_4 = [];

Doppler_all_1 = []; Doppler_all_2 = []; 
Doppler_all_3 = []; Doppler_all_4 = []; 
% N = 25; 
freq_point = 6; 
doppler_axis = linspace(-freq_point,freq_point,chirploops);
range_axis = flip(distance);
mean_signal_1 = []; mean_signal_2 = [];
mean_signal_3 = []; mean_signal_4 = [];
%i = 110
for n = 1 : 1
    u = n - 1;
    one_fr_sig_all_chirp_rec_1 = (R1(1+chirploops*u:chirploops+chirploops*u,:)+1i*I1(1+chirploops*u:chirploops+chirploops*u,:))';
    one_fr_sig_all_chirp_rec_2 = (R2(1+chirploops*u:chirploops+chirploops*u,:)+1i*I2(1+chirploops*u:chirploops+chirploops*u,:))';
    one_fr_sig_all_chirp_rec_3 = (R3(1+chirploops*u:chirploops+chirploops*u,:)+1i*I3(1+chirploops*u:chirploops+chirploops*u,:))';
    one_fr_sig_all_chirp_rec_4 = (R4(1+chirploops*u:chirploops+chirploops*u,:)+1i*I4(1+chirploops*u:chirploops+chirploops*u,:))';

    % run the FFT on the chirp signal along the 
    % range bins dimension and normalize.
    signal_fft_1 = flip(fft(one_fr_sig_all_chirp_rec_1, numADCSamples));
   
    signal_fft_2 = flip(fft(one_fr_sig_all_chirp_rec_2, numADCSamples));
   
    signal_fft_3 = (fft(one_fr_sig_all_chirp_rec_3, numADCSamples));
   
    signal_fft_4 = flip(fft(one_fr_sig_all_chirp_rec_4, numADCSamples));
   
%     range_get_1 = signal_fft_1(:,1); range_get_2 = signal_fft_2(:,1);
%     range_get_3 = signal_fft_3(:,1); range_get_4 = signal_fft_4(:,1);
    r1 = signal_fft_1'; r2 = signal_fft_2';
    r3 = signal_fft_3'; r4 = signal_fft_4';

    % Take the absolute value of FFT output
    signal_fft_1 = abs(signal_fft_1)./max(abs(signal_fft_1));
    signal_fft_2 = abs(signal_fft_2)./max(abs(signal_fft_2));
    signal_fft_3 = abs(signal_fft_3)./max(abs(signal_fft_3));
    signal_fft_4 = abs(signal_fft_4)./max(abs(signal_fft_4));
    
    %% 
      mean_signal_1 = (mean(signal_fft_1')');
%     mean_signal_2 = [mean_signal_2 mean(signal_fft_2')'];
%     mean_signal_3 = [mean_signal_3 mean(signal_fft_3')'];
%     mean_signal_4 = [mean_signal_4 mean(signal_fft_4')'];

%% 
%     signal = 1 * signal_fft_3(21:93,:);
%     signal_2 = 1 * signal_fft_3(150:220,:);
%     signal_fft_3_1 = [1*signal_fft_3(1:20,:); signal; 1*signal_fft_3(94:149,:); signal_2; 0.1*signal_fft_3(221:245,:); 1*signal_fft_3(246:256,:) ] ; %; signal_fft_3(101:256,:)];   

%%  
% plotting the range
    figure 
    plot(range_axis,signal_fft_3); 
    ylabel('Normalized Amplitude','FontSize', 17); 
    xlabel('Range (m)','FontSize', 17); axis([-25 25 0 2]); 
    axis tight; grid on; grid minor; 
    hold on; 

    %% RANGE DOPPLER MAP
% 2D FFT using the FFT size for both dimensions.
      signal_2d_fft_rec_1 = fftshift(fft2(one_fr_sig_all_chirp_rec_1, numADCSamples, chirploops));    
%     signal_2d_fft_rec_2 = fftshift(fft2(one_fr_sig_all_chirp_rec_2, numADCSamples, chirploops));    
%     signal_2d_fft_rec_3 = fftshift(fft2(one_fr_sig_all_chirp_rec_3, numADCSamples, chirploops));    
%     signal_2d_fft_rec_4 = fftshift(fft2(one_fr_sig_all_chirp_rec_4, numADCSamples, chirploops));   

%    
end

figure 
imagesc(range_axis,doppler_axis,mag2db(abs(signal_2d_fft_rec_1))); colorbar;

grid on; grid minor; xlabel('Range (m)','FontSize', 17); ylabel('Angle (degree)','FontSize', 17); zlabel('Amplitude');

% figure 
% imagesc(range_axis,doppler_axis,mag2db(abs(signal_2d_fft_rec_1))); colorbar;
% title('3D plot of AWR2243 with receiver 1');
% grid on; grid minor; xlabel('Range (m)'); ylabel('Velocity (m/s)'); zlabel('Amplitude');

%%
max_limit = 138; min_limit = 20;
mean_signal_1_cut = mean_signal_1([1:max_limit],:);
% mean_signal_signal_1_cut = 0.1+mean_signal_1([1:max_limit],:);
mean_signal_1_cut([1:min_limit],:) = 0;

% mean_signal_signal_2_cut = mean_signal_2([1:max_limit],:);
% % mean_signal_signal_2_cut = 0.1+mean_signal_2([1:max_limit],:);
% mean_signal_signal_2_cut([1:min_limit],:) = 0;
% 
% mean_signal_signal_3_cut = mean_signal_3([1:max_limit],:);
% % mean_signal_signal_3_cut = 0.1+mean_signal_3([1:max_limit],:);
% mean_signal_signal_3_cut([1:min_limit],:) = 0;
% 
% mean_signal_signal_4_cut = mean_signal_4([1:max_limit],:);
% % mean_signal_signal_4_cut = 0.1+mean_signal_4([1:max_limit],:);
% mean_signal_signal_4_cut([1:min_limit],:) = 0;

range1 = linspace(0,50,256);
range = range1(1:max_limit);

dopp = 1:length(mean_signal_1_cut(1,:));
dopp_g = dopp * 90 / length(dopp)  ;


%%
figure 
imagesc(range,dopp_g,mag2db(mean_signal_1_cut')); colorbar;

grid on; grid minor; xlabel('Range (m)','FontSize', 17); ylabel('Angle (degree)','FontSize', 17); zlabel('Amplitude');

%
figure(2); imagesc(range,dopp_g,mag2db(mean_signal_2_cut')); colorbar;
title('3D plot of rotating AWR2243 with receiver 2');
grid on; grid minor; xlabel('Range (m)'); ylabel('angle (degree)'); zlabel('Amplitude');
saveas(gcf,'HH1_rec_2.jpg');
saveas(gcf,'HH1_rec_2.fig');

saveas(gcf,'HH1_rec_4.jpg');
saveas(gcf,'HH1_rec_4.fig');

filename = 'HH1_rec_1.xlsx'; xlswrite(filename,mean_signal_signal_1_cut);
figure(3); imagesc(range,dopp_g,mag2db(mean_signal_signal_3_cut')); colorbar;
title('3D plot of rotating AWR2243 with receiver 3');
grid on; grid minor; xlabel('Range (m)'); ylabel('angle (degree)'); zlabel('Amplitude');
saveas(gcf,'HH1_rec_3.jpg');
saveas(gcf,'HH1_rec_3.fig');

figure(4); imagesc(range,dopp_g,mag2db(mean_signal_signal_4_cut')); colorbar;
title('3D plot of rotating AWR2243 with receiver 4');
grid on; grid minor; xlabel('Range (m)'); ylabel('angle (degree)'); zlabel('Amplitude');
filename = 'HH1_rec_2.xlsx'; xlswrite(filename,mean_signal_signal_2_cut);
filename = 'HH1_rec_3.xlsx'; xlswrite(filename,mean_signal_signal_3_cut);
filename = 'HH1_rec_4.xlsx'; xlswrite(filename,mean_signal_signal_4_cut);
