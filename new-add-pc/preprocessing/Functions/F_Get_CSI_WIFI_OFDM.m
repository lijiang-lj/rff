function [Get_CSI] = F_Get_CSI_WIFI_OFDM(Get_OFDM_Data)
    Get_CSI = zeros(3,64);
    Temp_1 = zeros(1,64);

    Get_Data_Index = 160;
    Temp_1(1,1:64) = Get_OFDM_Data(1,Get_Data_Index+32+1:Get_Data_Index+32+64);
    Temp_1 = fft(Temp_1);
    Get_CSI(1,1:64) = Temp_1(1,1:64);
 
    Get_Data_Index = Get_Data_Index +64;
    Temp_1(1,1:64) = Get_OFDM_Data(1,Get_Data_Index+32+1:Get_Data_Index+32+64);
    Temp_1 = fft(Temp_1);
    Get_CSI(2,1:64) = Temp_1(1,1:64);
    for n = 1:64
        Get_CSI(3,n) = (Get_CSI(1,n) + Get_CSI(2,n))/2;
    end

%     figure,
%     plot(1:64,abs(fftshift(Get_CSI(1,1:64))))
%     hold on
%     plot(1:64,abs(fftshift(Get_CSI(2,1:64))))