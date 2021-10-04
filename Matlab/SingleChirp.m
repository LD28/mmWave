clc 
clear all 
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This code reads bin file from ADC1000 and draw range,  % 
%range-velocity and Range-angle plots of A single Chirp %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%% % % Distance calculation using d=(c*f/(2*slope))
%The first bin in the FFT is DC (0 Hz), the second bin is Fs / N, where Fs 
%is the sample rate and N is the size of the FFT. The next bin is 2 * Fs / N. 
%To express this in general terms, the nth bin is n * Fs / N.

fdel_bin = (0:1:numADCSamples-1)*((5*10^6)/numADCSamples);
slope = 80*10^6;
distance = ((1.5*10^2)*fdel_bin)/slope;
ff = slope*2*distance/(3*10^8);

%% 
R1 = reshape(real_rece_1,numADCSamples,[])'; I1 = reshape(imag_rece_1,numADCSamples,[])';

I1= I1*1i;
    singleChirp = [R1(1,:)+ I1(1,:)];
    f_singleChirp= fft(singleChirp, numADCSamples);
    f_singleChirp_abs= abs(f_singleChirp./max(f_singleChirp));
plot( distance,f_singleChirp_abs);






