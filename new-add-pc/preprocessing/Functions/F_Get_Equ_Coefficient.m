function [Get_CSI_Equ] = F_Get_Equ_Coefficient(Get_CSI_Freq, CSI_Pattern)
    CSI_Length = length(CSI_Pattern);
    Get_CSI_Equ = zeros(1,CSI_Length);
    for n = 1:CSI_Length
        if(CSI_Pattern(1,n) == 0)
            Get_CSI_Equ(1,n) = 1;
        else
            Get_CSI_Equ(1,n) = CSI_Pattern(1,n)/Get_CSI_Freq(1,n);
        end
    end