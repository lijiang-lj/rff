function [Get_OFDM_Data] = F_Frequency_Offset_Compensationn(Get_OFDM_Data, Est_Freq_Compensate)
    Data_Length = length(Get_OFDM_Data);
    for n = 1:Data_Length
        Get_OFDM_Data(1,n) = Get_OFDM_Data(1,n) * exp(-1i*Est_Freq_Compensate*n);
    end