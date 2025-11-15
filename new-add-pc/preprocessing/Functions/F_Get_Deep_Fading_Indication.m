function [Deep_Fading_Indication] = F_Get_Deep_Fading_Indication(Get_CSI_Freq, CSI_Pattern)
    CSI_Length = length(CSI_Pattern);
    Get_CSI_Equ_Diff = zeros(CSI_Length,1);
    Get_CSI_Equ_Diff_2 = zeros(CSI_Length,1);
    Get_CSI_Equ = zeros(CSI_Length,1);
    
    Temp_1 = 0;
    for n = 1:CSI_Length
        Temp_1 = Temp_1 + abs(Get_CSI_Freq(n));
    end
    Temp_1 = Temp_1 / CSI_Length;
    for n = 1:CSI_Length
        if(CSI_Pattern(1,n) == 0)
        else
            Get_CSI_Equ(n) = abs(1 / Get_CSI_Freq(n) * Temp_1);
        end
    end
    
    for n = 1:CSI_Length-1
        if(CSI_Pattern(1,n) == 0 || CSI_Pattern(1,n+1) == 0)
        else
            Get_CSI_Equ_Diff(n) = abs(Get_CSI_Equ(n)) - abs(Get_CSI_Equ(n+1));
        end
    end
    for n = 1:CSI_Length-1
        Get_CSI_Equ_Diff_2(n) = abs(Get_CSI_Equ_Diff(n) - Get_CSI_Equ_Diff(n+1));
    end
    
    Temp_1 = 0;
    for n = 1:CSI_Length
        if(Get_CSI_Equ_Diff_2(n)>Temp_1)
            Temp_1 = Get_CSI_Equ_Diff_2(n);
        end
    end
    
    if(Temp_1 > 4)
        Deep_Fading_Indication = 1;
    else
        Deep_Fading_Indication = 0;
    end
    

