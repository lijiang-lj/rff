clear all
close all
clc

%% Parameters
addpath('Functions');
load('OFDM_STF_LTF.mat');
load('20200520_WiFi_Data_Waveform\P1\160212-003229.mat')

Ref_STF = zeros(1,128);
Ref_STF(1:128) = OFDM_STF(1:128);

Segment_Number = length(Store_Frame_Label);
Temp_1 = size(Store_Waveform);
Segment_Length = Temp_1(1,2);
Raw_Data = zeros(1,Segment_Length);
Get_Data_Length = 480;

for Segment_Process = 1:Segment_Number
    %% Pre Processing
    Raw_Data(1,1:Segment_Length) = Store_Waveform(Segment_Process,1:Segment_Length);
    Syn_Index_Coarse = 1;
    Corr_Find_Length = 100;
    [Syn_Index_Corr_Self] = F_Synchronization_Self_Corr(Raw_Data, Syn_Index_Coarse, 32, 64, 2, Corr_Find_Length);
    
    if(Syn_Index_Corr_Self>0 && Syn_Index_Corr_Self < 100)
        Get_Syn_Index = Syn_Index_Coarse + Syn_Index_Corr_Self - 4;
        
        %%% Get Data from File
        Get_OFDM_Data = zeros(1,Get_Data_Length);
        Get_OFDM_Data(1,1:Get_Data_Length) = Raw_Data(1,Get_Syn_Index+1:Get_Syn_Index+Get_Data_Length);
        
        %%% Estimate Signal Power
        [Get_Mean_Power] = F_Get_Signal_Power(Get_OFDM_Data, 160);
        
        %%% Estimate Frequency Offset
        [Est_Freq_Offset_Coarse] = F_Frequency_Offset_Estimation(Get_OFDM_Data, 0, 144, 16);
        [Get_OFDM_Data] = F_Frequency_Offset_Compensationn(Get_OFDM_Data, -Est_Freq_Offset_Coarse);
        
        Est_Frequency_Offset = Est_Freq_Offset_Coarse;
        
        %%% Estimate Channel Response
        [Get_CSI] = F_Get_CSI_WIFI_OFDM(Get_OFDM_Data);
        Get_CSI_Freq = zeros(1,64);
        Get_CSI_Freq(1,1:64) = Get_CSI(3,1:64);
        CSI_Pattern = circshift(LTF_Pattern,[0,32]);
        [Get_CSI_Equ] = F_Get_Equ_Coefficient(Get_CSI_Freq, CSI_Pattern);
        [Deep_Fading_Indication] = F_Get_Deep_Fading_Indication(Get_CSI_Freq, CSI_Pattern);
        
        %%% Get Data Symbols
        [Get_Data] = F_Get_Data_WIFI_OFDM(Get_OFDM_Data,Get_CSI_Equ);
        [Get_Data_Compensated,Compensation_Phase_Store,Est_EVM] = F_Get_WIFI_Data_Compensation(Get_Data);
        
        %%% Get Equlized Samples
        Get_STF_In = Get_OFDM_Data(1,1:160);
        [Get_STF_FFT_Equ] = F_Get_OFDM_Equ(Get_STF_In, Get_CSI_Equ);
        [Get_STF_FFT_Equ] = F_Get_Phase_Compensation(Get_STF_FFT_Equ, Ref_STF);
        
        %[Get_STF_FFT_Equ_Rebuild] = F_Frequency_Offset_Compensationn(Get_STF_FFT_Equ, Est_Frequency_Offset);
      
    end

end



