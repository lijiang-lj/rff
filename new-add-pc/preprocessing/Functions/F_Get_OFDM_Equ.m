function [Get_STF_Out] = F_Get_OFDM_Equ(Get_STF_In, Get_CSI_Equ)
    Temp_1 = zeros(1,64);
    Get_STF_Out = zeros(1,128);
    for OFDM_Symbol = 1:2
        Temp_1(1,1:64) = Get_STF_In(1,(OFDM_Symbol-1)*80+16+1:(OFDM_Symbol-1)*80+16+64);
        Temp_2 = fft(Temp_1);
        Temp_2 = Temp_2.*Get_CSI_Equ;
        Temp_2 = ifft(Temp_2);
        Get_STF_Out(1,(OFDM_Symbol-1)*64+1:OFDM_Symbol*64) = Temp_2(1,1:64);
    end
    