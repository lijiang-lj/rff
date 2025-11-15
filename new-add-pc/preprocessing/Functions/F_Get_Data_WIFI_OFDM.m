function [Get_Data] = F_Get_Data_WIFI_OFDM(Get_OFDM_Data,Get_CSI_Equ)
    Data_Length = length(Get_OFDM_Data);
    OFDM_Frames = floor(Data_Length/80) - 4;
    Get_Data_Length = OFDM_Frames * 53;
    if(Get_Data_Length>0)
        Get_Data = zeros(1,Get_Data_Length);
        Get_Data_Index = 0;
        Temp_1 = zeros(1,64);

        for n = 1:OFDM_Frames
            Temp_1(1,1:64) = Get_OFDM_Data(1,320+17+(n-1)*80:320+n*80);
            Temp_1 = fft(Temp_1);
            Temp_2 = Temp_1.*Get_CSI_Equ; 
            Temp_2 = circshift(Temp_2,[0,32]);

            Get_Data(1,Get_Data_Index+1:Get_Data_Index+53) = Temp_2(1,7:59);
    %         Get_Data(1,Get_Data_Index+27) = 0;
            Get_Data_Index = Get_Data_Index + 53;


    %         Get_Data(1,Get_Data_Index+1:Get_Data_Index+26) = Temp_2(1,2:27);
    %         Get_Data_Index = Get_Data_Index + 26;
    %         Get_Data(1,Get_Data_Index+1:Get_Data_Index+26) = Temp_2(1,39:64);
    %         Get_Data_Index = Get_Data_Index + 26;       
        end
    else
        Get_Data = 0;

    end