function [Get_Mean_Power] = F_Get_Signal_Power(Input_Data, Process_Length)
    Temp_1 = 0;
    for n = 1:Process_Length
        Temp_1 = Temp_1 + abs(Input_Data(1,n) * Input_Data(1,n));
    end
    Get_Mean_Power = Temp_1 / Process_Length;
    Get_Mean_Power = round(10*log10(Get_Mean_Power));
    
    