function [Syn_Index_Corr] = F_Synchronization_Self_Corr(Raw_Data, Syn_Start, Syn_Length, Syn_Interval, Syn_Period, Corr_Length)
    Data_Length = length(Raw_Data);
    Corr_Results = zeros(Corr_Length,1);
    
    if(Syn_Start+Corr_Length+Syn_Period*Syn_Interval+Syn_Length<Data_Length)
        %%% Get Correlation Results
        for Corr_Find_Index = 1:Corr_Length
            Temp_1 = 0;
            for n = 1:Syn_Length
                for m = 1:Syn_Period
                    Temp_1 = Temp_1 + Raw_Data(Syn_Start+(m-1)*Syn_Interval+Corr_Find_Index+n)*conj(Raw_Data(Syn_Start+m*Syn_Interval+Corr_Find_Index+n));
                end
            end
            Temp_1 = abs(Temp_1 / Syn_Length / Syn_Period);
            Corr_Results(Corr_Find_Index,1) = Temp_1;
        end
        %%% Get Maximal Correlation Index
        Temp_1 = 0;
        Temp_2 = 0;
        for n = 1:Corr_Length
            if(Corr_Results(n)>Temp_1)
                Temp_1 = Corr_Results(n);
                Temp_2 = n;
            end
        end
        Syn_Index_Corr = Temp_2;
        
    else
        Syn_Index_Corr = -1;
    end
    

