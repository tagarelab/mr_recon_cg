close all
clear
clc


%%


% EMI_Flag = 0;
EMI_Flag = 'EMIc_Off';
BaseLineCor_Flag = 'BaseLine_On';

INFO = [ 
         200 ... %1 Acq points
         50 ... %2 Echo Train Length
         0   ...                    %3 Number of Receiver Ch enabled on Rack 1
         1   ...                    %4 Number of Receiver Ch enabled on Rack 2
         0   ...                    %5 Number of Receiver Ch enabled on Rack 3
         0   ...                    %6 Number of Receiver Ch enabled on Rack 4
         1 ...                    %7 Number of echos at the end used in Self Correction (No NMR Signal)
         0 ... %8  #of echo to be averaged
         1 ...                      %9  #of pixles to be eliminated
         1 ...   %10 In ploting, Start from which echo
         5 ...                      %11 Parameter Starting Point
         10 ...                     %12 # of Calibration Steps
         1 ...                      %13 Parameter Resolution
       ];
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; 

load('GLR_0_rftime80us_90ampl_3.6_200pts_full_comb_07032024.mat');
DATA10 = raw_sig_all;

load('GLR_80_rftime80us_90ampl_3.6_200pts_comb_07032024.mat');
DATA80 = raw_sig_all;

len_train=10;

DATA10_reshape = reshape(DATA10, [INFO(1) INFO(2)]);
DATA80_reshape = reshape(DATA80, [INFO(1) INFO(2)]);
 

%%

val=40;
xx=linspace(-166665/1000,166665/1000,200);

idx_st=60; 
idx_end=141;

% % idx_st=76; 
% % idx_end=124;
% 
% idx_st=89; 
% idx_end=112;

% idx_st=96; 
% idx_end=105;

xx=xx(idx_st:idx_end); 


xMat = repmat(xx, val, 1); %// For plot3
xMat=xMat'; 

%// Define y values
y = 1:600:600*val;
yMat = repmat(y, size(xMat,1), 1); %//For plot3

%%

Z_80=[]; 


close all
for ii=1:val

    DATA10_reshape_reshape= DATA10_reshape(:,ii); 
    DATA80_reshape_reshape= DATA80_reshape(:,ii); 


    Projection_10 = ifftshift(fft(fftshift(DATA10_reshape_reshape)));
    Projection_80 = ifftshift(fft(fftshift(DATA80_reshape_reshape)));

    % Z_80=[Z_80  Projection_10(idx_st:idx_end)];
    Z_80=[Z_80  Projection_80(idx_st:idx_end)];
    


    
end


zMat = Z_80; 

%%
plot3(xMat, yMat, abs(zMat), 'black'); %// Make all traces blue
grid;
xlabel('kHz'); ylabel('time (us)');
%view(20,60); %// Adjust viewing angle so you can clearly see data
view(0,82);
set(gca, 'FontWeight', 'bold'); 
set(gca, 'FontSize', 12);
zlim([0 3*10^5]);



%% 
% xx=linspace(-166665/1000,166665/1000,200);
% for ii=1:5:45 
% 
%     DATA10_reshape_reshape= DATA10_reshape(:,ii); 
%     DATA80_reshape_reshape= DATA80_reshape(:,ii); 
% 
%     Projection_10 = ifftshift(fft(fftshift(DATA10_reshape_reshape)));
%     Projection_80 = ifftshift(fft(fftshift(DATA80_reshape_reshape)));
% 
%     figure(), hold on 
%     plot(xx,abs(Projection_10)); 
%     plot(xx,abs(Projection_80));
% 
%     legend({'no gradient', '80% gradient'}, 'Location', 'northwest');
% 
%     title(strcat('fft echo  = ', num2str(ii))); 
%     hold off
%     xlabel(' kHz '); 
%     set(gca, 'FontWeight', 'bold'); 
%     set(gca, 'FontSize', 12);
% end

