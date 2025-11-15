function [Get_Data_Out] = F_Get_Phase_Compensation(Get_Data_In, Ref_Data_In)
    Data_Length = length(Get_Data_In);
    Temp_1 = 0;
    Temp_2 = 0;
%     Angle_Store = zeros(1,Data_Length);
    for n = 1:Data_Length
        if(abs(Get_Data_In(n))>0.01)
            Temp_1 = Temp_1 + Get_Data_In(n)* conj(Ref_Data_In(n));
            Temp_2 = Temp_2 + 1;
        end
%         Angle_Store(n) =  Get_Data_In(n)* conj(Ref_Data_In(n));
    end
%     Angle_Store = angle(Angle_Store);
    if(Temp_2>0)
        Temp_1 = Temp_1 / Temp_2;
        Temp_1 = angle(Temp_1);
        Compensation_Factor = exp(-Temp_1*1i);
        Get_Data_Out = Get_Data_In * Compensation_Factor;
    else
        Get_Data_Out = Get_Data_In;
    end
    