
clc
close all
clear all
%% read file
% read .bin file
fid = fopen('1.6.bin','r');
adcData = fread(fid, 'int16');
fclose(fid);
fileSize = size(adcData, 1);
%% Check if the data can fit in 8 columns
 remaind = mod(fileSize,8);
 %if data is not divisable by 8, it means it cant be fit into 8 eight columns
 %Make data(Interleaved Data from AWR2243) over 8 columns.
if remaind ~= 0 
adcData =[ adcData;zeros(8-remaind,1)] ;
end
fileSize = length(adcData);
%% 
numADCSamples= 151;
fdel_bin = (0:1:numADCSamples-1)*((5*10^6)/numADCSamples);
slope = 80*10^6;
distance = ((1.5*10^2)*fdel_bin)/slope;
ff = slope*2*distance/(3*10^8);

%% Arrange Data according to LVDS lanes

lvds = reshape(adcData,8,[]);

lvds=lvds';
rx1= lvds(:,1)+lvds(:,5)*1i;
rx2= lvds(:,2)+lvds(:,6)*1i;
rx3= lvds(:,3)+lvds(:,7)*1i;
rx4= lvds(:,4)+lvds(:,8)*1i;

rx1_chirps = reshape(rx1,151,[])';
rx2_chirps = reshape(rx2,151,[])';
rx3_chirps = reshape(rx3,151,[])';
rx4_chirps = reshape(rx4,151,[])';


range= fft(rx1_chirps');
figure()

range_s = range(:,1:151*7);
%plot(distance,abs(range)./max(abs(range)));

rv = fft2(range_s);
figure()
imagesc(abs(rv)./max(abs(rv)))


