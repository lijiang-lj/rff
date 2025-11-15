function [Est_Freq_Offset] = F_Frequency_Offset_Estimation(Data_In, Syn_Start, Syn_Length, Syn_Interval)
    Mean_Thres = 0.2;
    Syn_Results_Raw = zeros(1,Syn_Length);
    for n = 1:Syn_Length
        Syn_Results_Raw(1,n) = Data_In(1,Syn_Start+n) * conj(Data_In(1,Syn_Start+Syn_Interval+n));
    end
    Mean_Power = mean(abs(Syn_Results_Raw));
    
    Syn_Results_Angle = angle(Syn_Results_Raw);
    Syn_Results_Raw = abs(Syn_Results_Raw);
    Temp_1 = 0;
    Temp_2 = 0;
    for n = 1:Syn_Length
        if(Syn_Results_Raw(1,n)>Mean_Power*Mean_Thres)
            Temp_2 = Temp_2 + Syn_Results_Angle(1,n);
            Temp_1 = Temp_1 + 1;
        end
    end
    Temp_3 = Temp_2 / Temp_1;
    Est_Freq_Offset = Temp_3 / Syn_Interval;
    
